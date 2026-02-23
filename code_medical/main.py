"""
Evolutionary Algorithm for Medical Image Dataset Distillation
-------------------------------------------------------------
NIH Chest X-ray 14 (6-class) + MedVAE latent space evolution.

Features:
  - Multi-GPU parallel evaluation (one individual per GPU concurrently)
  - Checkpoint/resume for SLURM auto-requeue (24h wall-time limit)
  - SIGUSR1 signal handler for graceful preemption
  - Always saves latest + best checkpoints

Usage:
  # Single GPU:
  python main.py --data_dir ./data --N 50 --IPC 50 --G 30

  # 8x A100 with auto-resume:
  python main.py --data_dir ./data --N 50 --IPC 50 --G 30 --num_gpus 8 --resume

Prerequisites:
  1. pip install -r requirements.txt
  2. Download Kaggle 224x224 NIH CXR-14 to --data_dir
  3. Run train_baselines.py first to populate baselines.json
"""
import os
import sys
import argparse
import json
import random
import signal
import time

import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import torchvision.utils as vutils

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ea.config import EvalCfg
from ea.data import get_nih_train_and_test, build_class_index, NUM_CLASSES
from ea.individual import Individual
from ea.stats import compute_real_stats
from ea.select import select_parents
from ea.utils import (
    load_medvae, encode_images, decode_latents, print_latent_stats,
    get_current_noise_std, crossover_z, mutate_z,
)
from ea.eval import evaluate_individual
from logger_utils import ExperimentLogger


# ═══════════════════════════════════════════════════════════════
#  Signal Handling (SLURM auto-requeue)
# ═══════════════════════════════════════════════════════════════
_exit_requested = False


def _handle_sigusr1(signum, frame):
    global _exit_requested
    print(f"\n[SIGNAL] SIGUSR1 at {time.strftime('%H:%M:%S')} "
          f"- will checkpoint and exit after current evaluation batch")
    _exit_requested = True


# ═══════════════════════════════════════════════════════════════
#  Checkpointing
# ═══════════════════════════════════════════════════════════════
def save_checkpoint(ckpt_dir, gen, pop, best_fitness, best_z,
                    rng_state, np_state, torch_state, logger_best, tag="latest"):
    """Atomic checkpoint save."""
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        'generation': gen,
        'population': [
            {'z': ind.z.cpu(), 'labels': ind.labels.cpu(),
             'fitness': ind.fitness, 'eval_info': ind.eval_info}
            for ind in pop
        ],
        'best_fitness': best_fitness,
        'best_z': best_z.cpu() if best_z is not None else None,
        'rng_state': rng_state,
        'np_rng_state': np_state,
        'torch_rng_state': torch_state,
        'logger_global_best': logger_best,
    }
    path = os.path.join(ckpt_dir, f"checkpoint_{tag}.pt")
    torch.save(ckpt, path + '.tmp')
    os.replace(path + '.tmp', path)
    print(f"[CKPT] Saved '{tag}' at gen {gen} (best_fit={best_fitness:.4f})")


def load_checkpoint(ckpt_dir, device, tag="latest"):
    """Load checkpoint. Returns (ckpt_dict, population) or (None, None)."""
    path = os.path.join(ckpt_dir, f"checkpoint_{tag}.pt")
    if not os.path.exists(path):
        return None, None
    print(f"[CKPT] Loading {path} ...")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    pop = []
    for d in ckpt['population']:
        ind = Individual(z=d['z'].to(device), labels=d['labels'].to(device),
                         fitness=d['fitness'])
        ind.eval_info = d['eval_info']
        pop.append(ind)
    print(f"[CKPT] Resumed: gen={ckpt['generation']}, best_fit={ckpt['best_fitness']:.4f}")
    return ckpt, pop


# ═══════════════════════════════════════════════════════════════
#  GPU Worker (multi-GPU parallel evaluation)
# ═══════════════════════════════════════════════════════════════
def _gpu_worker(gpu_id, task_queue, result_queue, worker_cfg):
    """
    Persistent worker on a specific GPU.
    Loads MedVAE + test data once, then evaluates individuals from the queue.
    Exits when it receives None (poison pill).
    """
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)

    mvae = load_medvae(device, model_name=worker_cfg['medvae_model'])

    _, test_norm, mean, std = get_nih_train_and_test(
        worker_cfg['data_dir'], max_per_class=worker_cfg['max_per_class'])
    test_loader = torch.utils.data.DataLoader(
        test_norm, batch_size=64, shuffle=False, num_workers=2)

    real_stats = {k: v.to(device) for k, v in worker_cfg['real_stats_cpu'].items()}
    eval_cfg = EvalCfg()
    baselines = worker_cfg['baselines']

    print(f"[Worker GPU:{gpu_id}] Ready", flush=True)

    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, z_cpu, labels_cpu = item
        ind = Individual(z=z_cpu.to(device), labels=labels_cpu.to(device))
        try:
            evaluate_individual(ind, mvae, eval_cfg, baselines,
                                test_loader, mean, std, device, real_stats)
            result_queue.put((idx, ind.fitness, ind.eval_info))
        except Exception as e:
            print(f"[Worker GPU:{gpu_id}] Error on ind {idx}: {e}", flush=True)
            result_queue.put((idx, float('-inf'), {
                'cvar': 0, 'score': float('-inf'),
                'per_model_scores': {}, 'best_norm_val': float('-inf'),
                'best_norm_model': 'error'}))

    print(f"[Worker GPU:{gpu_id}] Shutting down", flush=True)


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[SEED] {seed}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="EA Medical Image Distillation")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--medvae_model", type=str, default="medvae_4_1_2d")
    ap.add_argument("--N", type=int, default=50, help="Population size")
    ap.add_argument("--IPC", type=int, default=50, help="Images Per Class")
    ap.add_argument("--G", type=int, default=30, help="Generations")
    ap.add_argument("--mut_start", type=float, default=0.3)
    ap.add_argument("--mut_end", type=float, default=0.05)
    ap.add_argument("--max_per_class", type=int, default=3000)
    ap.add_argument("--baselines", type=str, default="baselines.json")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--num_gpus", type=int, default=1,
                    help="Number of GPUs for parallel evaluation")
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                    help="Directory for checkpoints (use persistent storage, NOT $TMPDIR)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint")
    args = ap.parse_args()

    # Register signal handler for SLURM preemption
    signal.signal(signal.SIGUSR1, _handle_sigusr1)

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    device = torch.device("cuda:0")
    print(f"GPUs available: {torch.cuda.device_count()}, using: {num_gpus}")

    # 1. Seed
    seed_everything(args.seed)

    # 2. Logger
    logger = ExperimentLogger(log_dir="logs", filename=f"log_seed{args.seed}.csv")

    # 3. Load baselines
    with open(args.baselines) as f:
        baselines = json.load(f)
    for mname in ["densenet121", "resnet50", "vit_small"]:
        if baselines["upper"].get(mname, 0.0) == 0.0:
            print(f"WARNING: baselines.json upper bound for {mname} is 0.0!")

    # 4. Load data
    train_raw, test_norm, mean, std = get_nih_train_and_test(
        args.data_dir, max_per_class=args.max_per_class)

    rng = random.Random(args.seed)
    real_stats = compute_real_stats(train_raw, 2000, rng)
    real_stats_cpu = {k: v.clone() for k, v in real_stats.items()}
    real_stats_dev = {k: v.to(device) for k, v in real_stats.items()}
    class_index = build_class_index(train_raw, num_classes=NUM_CLASSES)

    # 5. MedVAE (main process - for initialization + visualization)
    mvae = load_medvae(device, model_name=args.medvae_model)

    # 6. Resume or Initialize population
    start_gen = 0
    best_ever_fitness = float('-inf')
    best_ever_z = None
    pop = None

    if args.resume:
        ckpt, pop = load_checkpoint(args.checkpoint_dir, device)
        if ckpt is not None:
            start_gen = ckpt['generation'] + 1
            best_ever_fitness = ckpt['best_fitness']
            if ckpt['best_z'] is not None:
                best_ever_z = ckpt['best_z'].to(device)
            rng.setstate(ckpt['rng_state'])
            np.random.set_state(ckpt['np_rng_state'])
            torch.set_rng_state(ckpt['torch_rng_state'])
            logger.global_best_fit = ckpt['logger_global_best']

    if pop is None:
        pop = []
        print(f"\nInitializing population: N={args.N}, IPC={args.IPC}")
        for i in range(args.N):
            zs, lbls = [], []
            for c in range(NUM_CLASSES):
                pool = class_index[c]
                n = min(args.IPC, len(pool))
                idxs = rng.sample(pool, n)
                imgs = torch.stack([train_raw[j][0] for j in idxs])
                z = encode_images(mvae, imgs, device, batch_size=32)
                zs.append(z)
                lbls.append(torch.full((n,), c, dtype=torch.long, device=device))
            pop.append(Individual(z=torch.cat(zs), labels=torch.cat(lbls)))
            if i == 0:
                print_latent_stats(pop[0].z, label="Ind 0")
        z_std = pop[0].z.std().item()
        print(f"Latent std={z_std:.4f}  "
              f"mut=[{args.mut_start} -> {args.mut_end}]  "
              f"suggested=[{0.1*z_std:.4f} -> {0.02*z_std:.4f}]")

    if start_gen >= args.G:
        print(f"Already completed all {args.G} generations. Nothing to do.")
        return

    # 7. Start GPU workers (multi-GPU) or prepare single-GPU resources
    workers, task_q, result_q = [], None, None
    single_gpu_test_loader = None

    if num_gpus > 1:
        ctx = mp.get_context('spawn')
        task_q = ctx.Queue()
        result_q = ctx.Queue()
        worker_cfg = {
            'data_dir': args.data_dir,
            'medvae_model': args.medvae_model,
            'max_per_class': args.max_per_class,
            'baselines': baselines,
            'real_stats_cpu': real_stats_cpu,
            'seed': args.seed,
        }
        for gid in range(num_gpus):
            p = ctx.Process(target=_gpu_worker,
                            args=(gid, task_q, result_q, worker_cfg))
            p.start()
            workers.append(p)
        print(f"Started {num_gpus} GPU worker processes")
    else:
        single_gpu_test_loader = torch.utils.data.DataLoader(
            test_norm, batch_size=64, shuffle=False, num_workers=2)

    eval_cfg = EvalCfg()

    # ════════════════════════════════════════════════════════════
    #  Evolution Loop
    # ════════════════════════════════════════════════════════════
    try:
        for gen in range(start_gen, args.G):
            gen_start = time.time()
            print(f"\n{'='*60}")
            print(f"  Generation {gen}/{args.G-1}  |  {time.strftime('%H:%M:%S')}")
            print(f"{'='*60}")

            # ─── A. Evaluate unevaluated individuals ───
            to_eval = [(i, ind) for i, ind in enumerate(pop)
                       if ind.fitness == float('-inf')]

            if num_gpus > 1 and to_eval:
                # Multi-GPU: submit all work items, then collect all results
                for idx, ind in to_eval:
                    task_q.put((idx, ind.z.cpu(), ind.labels.cpu()))

                for count in range(len(to_eval)):
                    idx, fitness, eval_info = result_q.get()
                    pop[idx].fitness = fitness
                    pop[idx].eval_info = eval_info
                    logger.log_individual(
                        gen=gen, ind_id=idx, fitness=fitness,
                        scores_dict=eval_info.get('per_model_scores', {}),
                        best_norm_curr=eval_info.get('best_norm_val', 0),
                        best_norm_model=eval_info.get('best_norm_model', 'N/A'),
                    )
                    print(f"  [{count+1}/{len(to_eval)}] Ind {idx}: "
                          f"Fit={fitness:.4f} | "
                          f"Top={eval_info.get('best_norm_val', 0):.4f} "
                          f"({eval_info.get('best_norm_model', 'N/A')})")

            elif to_eval:
                # Single-GPU evaluation
                for idx, ind in to_eval:
                    fit = evaluate_individual(
                        ind, mvae, eval_cfg, baselines,
                        single_gpu_test_loader, mean, std, device, real_stats_dev)
                    info = ind.eval_info
                    logger.log_individual(
                        gen=gen, ind_id=idx, fitness=fit,
                        scores_dict=info['per_model_scores'],
                        best_norm_curr=info['best_norm_val'],
                        best_norm_model=info['best_norm_model'],
                    )
                    print(f"  Ind {idx}: Fit={fit:.4f} | "
                          f"Top={info['best_norm_val']:.4f} ({info['best_norm_model']})")
                    if _exit_requested:
                        break

            logger.plot_curves()

            # ─── Sort & track best ───
            pop.sort(key=lambda x: x.fitness, reverse=True)
            gen_best = pop[0]
            elapsed = time.time() - gen_start
            print(f"Gen {gen} Best: {gen_best.fitness:.4f}  "
                  f"(elapsed: {elapsed:.0f}s)")

            if gen_best.fitness > best_ever_fitness:
                best_ever_fitness = gen_best.fitness
                best_ever_z = gen_best.z.clone()
                torch.save(best_ever_z.cpu(), "best_z.pt")
                save_checkpoint(
                    args.checkpoint_dir, gen, pop, best_ever_fitness,
                    best_ever_z, rng.getstate(), np.random.get_state(),
                    torch.get_rng_state(), logger.global_best_fit, tag="best")

            # ─── Visualize best individual ───
            with torch.no_grad():
                vis_z = gen_best.z[:64].to(device)
                imgs = decode_latents(mvae, vis_z, device, batch_size=32)
                vutils.save_image(imgs.repeat(1, 3, 1, 1),
                                  f"logs/best_img_gen_{gen}.png", nrow=8, padding=2)

            # ─── Save latest checkpoint (every generation) ───
            save_checkpoint(
                args.checkpoint_dir, gen, pop, best_ever_fitness,
                best_ever_z, rng.getstate(), np.random.get_state(),
                torch.get_rng_state(), logger.global_best_fit, tag="latest")

            # ─── Check exit signal ───
            if _exit_requested:
                print("[EXIT] Graceful shutdown (SIGUSR1). Checkpoint saved.")
                break

            if gen == args.G - 1:
                break

            # ─── B. Reproduce ───
            elite_size = max(1, int(args.N * 0.2))
            elites = pop[:elite_size]
            parents = select_parents(
                pop, num_pairs=args.N - elite_size, temperature=1.0, rng=rng)
            current_std = get_current_noise_std(
                gen, args.G, args.mut_start, args.mut_end)

            next_gen = []
            for p1, p2 in parents:
                child_z = crossover_z(p1.z, p2.z)
                child_z = mutate_z(child_z, current_std)
                next_gen.append(Individual(z=child_z, labels=p1.labels.clone()))

            pop = elites + next_gen

    finally:
        # ─── Shutdown workers ───
        if workers:
            print("Shutting down GPU workers...")
            for _ in workers:
                task_q.put(None)
            for p in workers:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()

    print("\nEvolution Finished.")


if __name__ == "__main__":
    main()
