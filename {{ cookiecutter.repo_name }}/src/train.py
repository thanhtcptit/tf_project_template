import os
import time
import shutil
import collections

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from tqdm import tqdm
from pprint import pprint

from src.utils import *
from src.models.base import BaseModel

tf.get_logger().setLevel('INFO')


def create_tf_dataset(data, batch_size, is_train=False):
    tf_dataset = tf.data.Dataset.from_tensor_slices(data)
    if is_train:
        tf_dataset = tf_dataset.shuffle(buffer_size=batch_size * 100)
    tf_dataset = tf_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return tf_dataset


def evaluate(model, test_dataset, trainer_config):
    metrics = []
    if trainer_config["loss_fn"]["type"] in CLASSIFICATION_LOSSES:
        metrics += [
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.PrecisionAtRecall(recall=0.8)
        ]

    loss_fn = loss_fn = build_loss_fn(trainer_config["loss_fn"])
    mean_loss = keras.metrics.Mean()
    for batch in test_dataset:
        preds = model(batch[0], training=False)
        y_true = tf.expand_dims(batch[1], -1)
        for metric in metrics:
            metric.update_state(y_true, preds)
        loss = loss_fn(y_true, preds)
        mean_loss.update_state(loss)

    results = [mean_loss.result().numpy()]
    for metric in metrics:
        results.append(metric.result().numpy().tolist())
        metric.reset_states()
    return results


def train(config_path, dataset_path, checkpoint_dir, recover=False, force=False):
    config = Params.from_file(config_path)
    config["dataset_path"] = dataset_path
    model_config = config["model"]
    trainer_config = config["trainer"]
    pprint(config.as_dict())

    if not checkpoint_dir:
        dataset_name = get_basename(dataset_path)
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        checkpoint_dir = os.path.join("train_logs", dataset_name, config_name)
    if os.path.exists(checkpoint_dir):
        if force:
            shutil.rmtree(checkpoint_dir)
        else:
            raise ValueError(f"{checkpoint_dir} already existed")
    weight_dir = os.path.join(checkpoint_dir, "checkpoints")
    os.makedirs(weight_dir, exist_ok=True)
    save_json(os.path.join(checkpoint_dir, "config.json"), config.as_dict())
    print("Train log: ", checkpoint_dir)

    train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(dataset_path, "val.csv"))

    train_dataset = create_tf_dataset(train_df, trainer_config["batch_size"], is_train=True)
    val_dataset = create_tf_dataset(val_df, trainer_config["batch_size"])

    model = BaseModel.from_params(model_config).build_graph()
    if recover:
        model.load_weights(weight_dir)

    loss_fn = build_loss_fn(trainer_config["loss_fn"])
    optimizer = build_optimizer(trainer_config["optimizer"])

    # Custom training loop
    # log_file = os.path.join(checkpoint_dir, "log.txt")
    # logger = Logger(log_file, stdout=False)
    # logger.log(f"\n=======================================\n")

    # @tf.function
    # def train_step(x, y, train_user_emb=True, train_item_emb=True):
    #     with tf.GradientTape() as tape:
    #         preds = model(x, training=True)
    #         loss_value = loss_fn(tf.expand_dims(y, -1), preds)

    #     trainable_weights = []
    #     if train_user_emb:
    #         trainable_weights.append(model.trainable_weights[0])
    #     if train_item_emb:
    #         trainable_weights.append(model.trainable_weights[1])
    #     grads = tape.gradient(loss_value, trainable_weights)
    #     if "grad_clip" in trainer_config:
    #         grads = [(tf.clip_by_value(grad, clip_value_min=trainer_config["grad_clip"]["min_value"],
    #                                    clip_value_max=trainer_config["grad_clip"]["max_value"]))
    #                                    for grad in grads]
    #     optimizer.apply_gradients(zip(grads, trainable_weights))
    #     return loss_value

    # loss_fn_name = trainer_config["loss_fn"]["type"]
    # print(("\n" + " %10s " * 7) % (loss_fn_name, f"val_{loss_fn_name}", "val_acc",
    #                              "val_p", "val_r", "val_p@r0.8", "iter"))
    # logger.log(("\n" + " %10s " * 7) % ("iter", loss_fn_name, f"val_{loss_fn_name}", "val_acc",
    #                                     "val_p", "val_r", "val_p@r0.8"))
    # results = evaluate(model, val_dataset, trainer_config)
    # pbar = tqdm(enumerate(train_dataset))
    # pbar.set_description(("%10.4g" * (1 + len(results))) % (0, *results))

    # best_loss = float("inf")
    # for step, batch in pbar:
    #     if step > trainer_config["num_steps"]:
    #         pbar.close()
    #         break
    #     loss_value = train_step(batch[0], batch[1])
    #     if step % trainer_config["display_step"] == 0:
    #         pbar.set_description(("%10.4g" * (1 + len(results))) % (loss_value, *results))

    #     if (step + 1) % trainer_config["save_step"] == 0:
    #         model.save_weights(os.path.join(weight_dir, "latest.ckpt"))

    #     if (step + 1) % trainer_config["validate_step"] == 0:
    #         results = evaluate(model, val_dataset, trainer_config)
    #         pbar.set_description(("%10.4g" * (1 + len(results))) % (loss_value, *results))
    #         logger.log(("\n" + "%10.4g" * (2 + len(results))) % (step + 1, loss_value, *results))
    #         if best_loss > results[0]:
    #             best_loss = results[0]
    #             model.save_weights(os.path.join(weight_dir, "best.ckpt"))

    # Trainer API
    callbacks = []
    if "callbacks" in trainer_config:
        for callback in trainer_config["callbacks"]:
            if callback["type"] == "model_checkpoint":
                callback["filepath"] = os.path.join(weight_dir, "best.ckpt")
            elif callback["type"] == "logging":
                callback["filename"] = os.path.join(checkpoint_dir, "log.csv")
            callbacks.append(build_callback_fn(callback))

    metrics = []
    if trainer_config["loss_fn"]["type"] in CLASSIFICATION_LOSSES:
        metrics += [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics)
    model.fit(
        train_dataset, validation_data=val_dataset, epochs=trainer_config["num_epochs"],
        callbacks=callbacks)

    metrics = model.evaluate(val_dataset)
    print(metrics)
    return metrics


def test(checkpoint_dir, test_dataset_path):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    dataset_path = config["dataset_path"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    if not test_dataset_path:
        test_dataset_path = os.path.join(dataset_path, "val.csv")
    test_df = pd.read_csv(test_dataset_path)
    test_dataset = create_tf_dataset(test_df, trainer_config["batch_size"], is_train=True)

    model = BaseModel.from_params(model_config).build_graph()
    model.load_weights(os.path.join(checkpoint_dir, "checkpoints/best.ckpt"))

    metrics = []
    if trainer_config["loss_fn"]["type"] in CLASSIFICATION_LOSSES:
        metrics += [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)]
    model.compile(metrics=metrics)
    metrics = model.evaluate(test_dataset)
    print(metrics)
    return metrics


def hyperparams_search(config_file, dataset_path, test_dataset_path, num_trials=50, force=False):
    import optuna
    from optuna.integration import TFKerasPruningCallback

    def objective(trial):
        tf.keras.backend.clear_session()

        dataset_name = get_basename(dataset_path)
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        config = load_json(config_file)
        hyp_config = config["hyp"]
        for k, v in hyp_config.items():
            k_list = k.split(".")
            d = config
            for i in range(len(k_list) - 1):
                d = d[k_list[i]]
            if v["type"] == "int":
                val = trial.suggest_int(k, v["range"][0], v["range"][1])
            elif v["type"] == "float":
                val = trial.suggest_float(k, v["range"][0], v["range"][1], log=v.get("log", False))
            elif v["type"] == "categorical":
                val = trial.suggest_categorical(k, v["values"])
            d[k_list[-1]] = val
            config_name += f"_{k_list[-1]}-{val}"

        config.pop("hyp")
        checkpoint_dir = f"/tmp/{dataset_name}/{config_name}"
        trial_config_file = os.path.join(f"/tmp/hyp_{get_current_time_str()}.json")
        save_json(trial_config_file, config)

        best_val = train(trial_config_file, dataset_path, checkpoint_dir, force=force)[0]
        if test_dataset_path:
            best_val = test(checkpoint_dir, test_dataset_path)[0]
        return best_val

    study = optuna.create_study(study_name="hyp", direction="maximize")
    study.optimize(objective, n_trials=num_trials, gc_after_trial=True,
                   catch=(tf.errors.InvalidArgumentError,))
    print("Number of finished trials: ", len(study.trials))

    df = study.trials_dataframe()
    print(df)

    print("Best trial:")
    trial = study.best_trial

    print(" - Value: ", trial.value)
    print(" - Params: ")
    for key, value in trial.params.items():
        print("  - {}: {}".format(key, value))
