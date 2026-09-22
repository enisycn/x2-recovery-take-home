#!/usr/bin/env python3
"""Distill successful legacy recovery rollouts into the HumanUP RMA actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from x2_recovery_isaac.agents.humanup_history_model import HumanUpHistoryActor


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--dataset", type=Path, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=1.0e-4)
parser.add_argument("--seed", type=int, default=127)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()


def make_actor(sample: torch.Tensor, device: torch.device) -> HumanUpHistoryActor:
    dummy = TensorDict({"policy": sample[:1].to(device)}, batch_size=[1])
    return HumanUpHistoryActor(
        dummy,
        {"actor": ["policy"]},
        "actor",
        31,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.2},
        proprio_dim=98,
        history_length=10,
        history_latent_dim=20,
    ).to(device)


def predict(actor: HumanUpHistoryActor, observations: torch.Tensor) -> torch.Tensor:
    return actor(TensorDict({"policy": observations}, batch_size=[observations.shape[0]]))


def mse(actor: HumanUpHistoryActor, observations: torch.Tensor, actions: torch.Tensor, batch: int) -> float:
    total = 0.0
    count = 0
    actor.eval()
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch):
            target = actions[start : start + batch]
            loss = torch.square(predict(actor, observations[start : start + batch]) - target).mean()
            total += float(loss.item()) * target.shape[0]
            count += target.shape[0]
    return total / count


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    obs_parts = []
    action_parts = []
    sources = []
    for dataset_path in args.dataset:
        path = dataset_path.expanduser().resolve(strict=True)
        with np.load(path) as data:
            obs_parts.append(torch.from_numpy(data["observations"].astype(np.float32)))
            action_parts.append(torch.from_numpy(data["actions"].astype(np.float32)))
            sources.append({"path": str(path), "samples": int(data["actions"].shape[0])})
    observations = torch.cat(obs_parts)
    actions = torch.cat(action_parts)
    if observations.shape[1] != 1148 or actions.shape[1] != 31:
        raise ValueError(f"Unexpected dataset shapes: {observations.shape}, {actions.shape}")

    permutation = torch.randperm(observations.shape[0])
    validation_count = max(1, observations.shape[0] // 10)
    validation_ids = permutation[:validation_count]
    training_ids = permutation[validation_count:]
    train_obs = observations[training_ids].to(device)
    train_actions = actions[training_ids].to(device)
    val_obs = observations[validation_ids].to(device)
    val_actions = actions[validation_ids].to(device)

    source_checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    checkpoint = torch.load(source_checkpoint, map_location=device, weights_only=False)
    actor = make_actor(observations, device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    initial_train_mse = mse(actor, train_obs, train_actions, args.batch_size)
    initial_validation_mse = mse(actor, val_obs, val_actions, args.batch_size)

    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    actor.train()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    epoch_losses = []
    for _ in range(args.epochs):
        order = torch.randperm(train_obs.shape[0], generator=generator, device="cpu").to(device)
        running = 0.0
        seen = 0
        for start in range(0, train_obs.shape[0], args.batch_size):
            ids = order[start : start + args.batch_size]
            target = train_actions[ids]
            prediction = predict(actor, train_obs[ids])
            loss = torch.square(prediction - target).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * target.shape[0]
            seen += target.shape[0]
        epoch_losses.append(running / seen)

    final_train_mse = mse(actor, train_obs, train_actions, args.batch_size)
    final_validation_mse = mse(actor, val_obs, val_actions, args.batch_size)
    checkpoint["actor_state_dict"] = actor.state_dict()
    checkpoint["infos"] = {
        "behavioral_cloning": {
            "sources": sources,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "initial_validation_mse": initial_validation_mse,
            "final_validation_mse": final_validation_mse,
        }
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    report = {
        "source_checkpoint": str(source_checkpoint),
        "output_checkpoint": str(output),
        "sources": sources,
        "training_samples": int(train_obs.shape[0]),
        "validation_samples": int(val_obs.shape[0]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "initial_train_mse": initial_train_mse,
        "initial_validation_mse": initial_validation_mse,
        "final_train_mse": final_train_mse,
        "final_validation_mse": final_validation_mse,
        "last_epoch_mse": epoch_losses[-1],
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
