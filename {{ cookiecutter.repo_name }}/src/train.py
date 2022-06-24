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

from src.utils.train_utils import *
from src.models.base import BaseModel
from src.utils import Params, save_json, save_txt, load_json, get_current_time_str, Logger

tf.get_logger().setLevel('INFO')


def create_tf_dataset(data, batch_size, is_train=False):
    tf_dataset = tf.data.Dataset.from_tensor_slices(data)
    if is_train:
        tf_dataset = tf_dataset.shuffle(buffer_size=batch_size * 100)
    tf_dataset = tf_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return tf_dataset


def evaluate(model, test_dataset, trainer_config):
    if trainer_config["loss_fn"]["type"] in ["bce", "wbce", "cce", "focal"]:
        metrics = [
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.PrecisionAtRecall(recall=0.8)
        ]
    else:
        metrics = []
    loss_fn = get_loss_fn(trainer_config["loss_fn"]["type"])(**trainer_config["loss_fn"].get("params", {}))
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


def train(config_path, checkpoint_dir, recover=False, force=False):
    config = Params.from_file(config_path)
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]
    pprint(config.as_dict())

    if not checkpoint_dir:
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        checkpoint_dir = os.path.join("train_logs", config_name)
    if os.path.exists(checkpoint_dir):
        if force:
            shutil.rmtree(checkpoint_dir)
        elif not recover:
            raise ValueError(f"{checkpoint_dir} already exists!")
    weight_dir = os.path.join(checkpoint_dir, "checkpoints")
    os.makedirs(weight_dir, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(checkpoint_dir, "config.json"))
    print("Train log: ", checkpoint_dir)

    train_df = pd.read_csv(data_config["path"]["train"])
    val_df = pd.read_csv(data_config["path"]["val"])

    train_dataset = create_tf_dataset(train_df, trainer_config["batch_size"], is_train=True)
    val_dataset = create_tf_dataset(val_df, trainer_config["batch_size"])

    model = BaseModel.from_params(model_config).build_graph()
    if recover:
        model.load_weights(weight_dir)

    optimizer = get_optimizer(trainer_config["optimizer"]["type"])(**trainer_config["optimizer"].get("params", {}))
    loss_fn = get_loss_fn(trainer_config["loss_fn"]["type"])(**trainer_config["loss_fn"].get("params", {}))

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
            if "params" not in callback:
                callback["params"] = {}
            if callback["type"] == "model_checkpoint":
                callback["params"]["filepath"] = os.path.join(weight_dir, "best.ckpt")
            elif callback["type"] == "logging":
                callback["params"]["filename"] = os.path.join(checkpoint_dir, "log.csv")
            callbacks.append(get_callback_fn(callback["type"])(**callback["params"]))
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
    model.fit(
        train_dataset, validation_data=val_dataset, epochs=trainer_config["num_epochs"],
        callbacks=callbacks)

    metrics = model.evaluate(val_dataset)
    print(metrics)
    return metrics


def test(checkpoint_dir, dataset_path):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    if not dataset_path:
        dataset_path = data_config["path"]["val"]
    test_df = pd.read_csv(dataset_path)
    test_dataset = create_tf_dataset(test_df, trainer_config["batch_size"], is_train=True)

    model = BaseModel.from_params(model_config).build_graph()
    model.load_weights(os.path.join(checkpoint_dir, "checkpoints/best.ckpt"))
    model.compile(
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
    metrics = model.evaluate(test_dataset)
    print(metrics)
    return metrics


def hyperparams_search(config_file, dataset_path, num_trials=50, force=False):
    import optuna
    from optuna.integration import TFKerasPruningCallback

    def objective(trial):
        tf.keras.backend.clear_session()

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
        checkpoint_dir = f"/tmp/{config_name}"
        trial_config_file = os.path.join(f"/tmp/hyp_{get_current_time_str()}.json")
        save_json(trial_config_file, config)

        best_val = train(trial_config_file, checkpoint_dir, force=force)[0]
        if dataset_path:
            best_val = test(checkpoint_dir, dataset_path)[0]
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
