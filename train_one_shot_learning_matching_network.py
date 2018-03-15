#Copyright 2018 UNIST under XAI Project supported by Ministry of Science and ICT, Korea

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   https://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from one_shot_learning_network import *
from experiment_builder import ExperimentBuilder
import tensorflow.contrib.slim as slim
import time
import data as dataset
import tqdm
from storage import *

tf.reset_default_graph()

# Experiment Setup
batch_size = 5
fce = True
classes_per_set = 2
samples_per_class = 1
continue_from_epoch = -1  # use -1 to start from scratch
epochs = 10
k = 1

logs_path = "one_shot_outputs_ADNI/"
experiment_name = "one_shot_ADNI_{}fold_{}_{}".format(k, samples_per_class, classes_per_set)

# Experiment builder

# data = dataset.OmniglotNShotDataset(batch_size=batch_size, classes_per_set=classes_per_set,
#                                     samples_per_class=samples_per_class)

data = dataset.ADNIDataset(k=k, batch_size=batch_size, classes_per_set=classes_per_set, samples_per_class=samples_per_class)



experiment = ExperimentBuilder(data)
one_shot_omniglot, losses, c_error_opt_op, init = experiment.build_experiment(batch_size, classes_per_set, samples_per_class, fce)

total_epochs = 100
total_train_batches = 5
total_val_batches = 2
total_test_batches = 2

save_statistics(experiment_name, ["epoch", "train_c_loss", "train_c_accuracy", "val_loss", "val_accuracy",
                                  "test_c_loss", "test_c_accuracy"])

# summary_path = "/summary/%d" % (int(time.time()))
# Experiment initialization and running

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    # summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
    if continue_from_epoch != -1: #load checkpoint if needed
        checkpoint = "saved_models/{}_{}.ckpt".format(experiment_name, continue_from_epoch)
        variables_to_restore = []
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(var)
            variables_to_restore.append(var)

        tf.logging.info('Fine-tuning from %s' % checkpoint)

        fine_tune = slim.assign_from_checkpoint_fn(
            checkpoint,
            variables_to_restore,
            ignore_missing_vars=True)
        fine_tune(sess)

    best_val = 0.
    with tqdm.tqdm(total=total_epochs) as pbar_e:
        for e in range(0, total_epochs):
            total_c_loss, total_accuracy = experiment.run_training_epoch(total_train_batches=total_train_batches,
                                                                         sess=sess)
            # tf.summary.scalar("loss_train", total_c_loss)
            # tf.summary.scalar("acc_train", total_accuracy)


            print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

            total_val_c_loss, total_val_accuracy = experiment.run_validation_epoch(
                total_val_batches=total_val_batches,
                sess=sess)
            # tf.summary.scalar("loss_val", total_val_c_loss)
            # tf.summary.scalar("loss_val", total_val_accuracy)

            print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

            if total_val_accuracy >= best_val: #if new best val accuracy -> produce test statistics
                best_val = total_val_accuracy
                total_test_c_loss, total_test_accuracy = experiment.run_testing_epoch(
                    total_test_batches=total_test_batches,
                    sess=sess)
                print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
            else:
                total_test_c_loss = -1
                total_test_accuracy = -1

            save_statistics(experiment_name,
                            [e, total_c_loss, total_accuracy, total_val_c_loss, total_val_accuracy, total_test_c_loss,
                             total_test_accuracy])

            save_path = saver.save(sess, "saved_models/{}_{}.ckpt".format(experiment_name, e))
            pbar_e.update(1)
