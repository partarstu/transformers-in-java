/*
 * Copyright 2023 Taras Paruta
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package org.tarik.train;

import com.google.common.util.concurrent.AtomicDouble;
import com.google.gson.Gson;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tarik.core.network.models.transformer.AbstractTransformerSameDiffModel;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Runtime.getRuntime;
import static java.lang.String.format;
import static java.nio.file.Files.exists;
import static java.nio.file.Files.move;
import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static java.time.Instant.now;
import static java.util.concurrent.Executors.newSingleThreadScheduledExecutor;
import static java.util.concurrent.TimeUnit.MINUTES;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.bytedeco.javacpp.Pointer.availablePhysicalBytes;
import static org.bytedeco.javacpp.Pointer.physicalBytes;
import static org.nd4j.linalg.factory.Nd4j.getMemoryManager;
import static org.tarik.utils.CommonUtils.*;

public class CommonTrainer {
    private static final Logger LOG = LoggerFactory.getLogger(CommonTrainer.class);
    protected static final String MAX_JVM_MEMORY = System.getenv().getOrDefault("max_jvm_memory", "60G");
    protected static final String DEALLOCATOR_THREADS = System.getenv().getOrDefault("DEALLOCATOR_SERVICE_GC_THREADS", "8");

    protected static final AtomicBoolean savingInProgress = new AtomicBoolean(false);
    protected static final AtomicBoolean lastSaveWasSuccessful = new AtomicBoolean(false);

    protected static void prepareEnvironment() {
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", MAX_JVM_MEMORY);
        System.setProperty("org.nd4j.autodiff.samediff.cache", "false");
        System.setProperty("org.nd4j.linalg.api.ops.udf.packages", "org.tarik.core.neural.network.custom_ops");
        System.setProperty("org.nd4j.deallocator.threads", DEALLOCATOR_THREADS);

        getMemoryManager().setAutoGcWindow(10000);
        getMemoryManager().togglePeriodicGc(true);
    }

    protected synchronized static <T extends AbstractTransformerSameDiffModel<T>> void saveModel(T model, Path modelPath,
                                                                                                Path backupPath) {
        savingInProgress.getAndSet(true);
        Instant saveStart = now();
        String message = "Saved a %d MB model ";
        try {
            if (exists(modelPath) && lastSaveWasSuccessful.get()) {
                var timestamp =
                        getFileLastModifiedDateTime(modelPath).map(String::valueOf).orElse("unknown_timestamp");
                move(modelPath, backupPath, REPLACE_EXISTING);
                LOG.info("Updated a backup with a newer model version from {}", timestamp);
            } else {
                LOG.warn("Update of a backup with a newer model version was skipped due to unsuccessful previous save");
            }

            model.save(modelPath);
            lastSaveWasSuccessful.set(true);
            long modelSize = Files.size(modelPath) / 1024 / 1024;
            LOG.info("{} in {} seconds\n", format(message, modelSize), getDurationInSeconds(saveStart));
        } catch (Throwable t) {
            lastSaveWasSuccessful.set(false);
            LOG.error("Saving a model failed. ", t);
            try {
                throw t;
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        } finally {
            savingInProgress.getAndSet(false);
        }
    }

    protected static void startMemoryProfiling(int frequency, TimeUnit timeUnit) {
        ScheduledExecutorService scheduledExecutorService = newSingleThreadScheduledExecutor();
        Supplier<Double> availableMemProvider = () -> (double) availablePhysicalBytes() / 1024 / 1024 / 1024;
        AtomicDouble maxMemoryGb = new AtomicDouble(0);
        AtomicDouble availableMemoryGb = new AtomicDouble(availableMemProvider.get());

        scheduledExecutorService.scheduleAtFixedRate(() -> {
            double usedMemory = (double) physicalBytes() / 1024 / 1024 / 1024;
            double availableMemory = availableMemProvider.get();
            maxMemoryGb.set(max(maxMemoryGb.get(), usedMemory));
            availableMemoryGb.set(min(availableMemoryGb.get(), availableMemory));
        }, 0, 1, SECONDS);
        scheduledExecutorService.scheduleAtFixedRate(() -> {
            try {
                LOG.info("Memory taken : {} GB,  free : {} GB",
                        format("%.1f", maxMemoryGb.get()), format("%.1f", availableMemoryGb.get()));
                maxMemoryGb.getAndSet(0);
                availableMemoryGb.getAndSet(availableMemProvider.get());
            } catch (Exception e) {
                LOG.info("Couldn't log memory. Original error: {}", e.getMessage());
            }
        }, 0, frequency, timeUnit);
    }

    protected static void startMemoryProfiling(int freqInMinutes) {
        startMemoryProfiling(freqInMinutes, MINUTES);
    }

    protected static <T extends AbstractTransformerSameDiffModel<T>> void loadModelVariablesAndUpdater(T model, Path modelPath,
                                                                                                       Path backupPath,
                                                                                                       boolean loadUpdater)
            throws IOException {
        Instant loadStart = now();
        LOG.debug("Started loading model");
        try {
            model.loadModelDataFromFile(modelPath, loadUpdater);
            long modelSize = Files.size(modelPath) / 1024 / 1024;
            LOG.info("Loaded a {} MB model in {} seconds \n", modelSize, getDurationInSeconds(loadStart));
        } catch (Exception e) {
            if (exists(backupPath)) {
                LOG.warn("Couldn't load the last saved version of the model. Original exception message: {}", e.getMessage());
                var backupTimestamp = getFileLastModifiedDateTime(backupPath).map(String::valueOf).orElse("unknown_timestamp");
                LOG.info("Trying to load the model's back-up from {}", backupTimestamp);
                loadStart = now();
                model.loadModelDataFromFile(backupPath, loadUpdater);
                long modelSize = Files.size(backupPath) / 1024 / 1024;
                LOG.info("Loaded a {} MB model's back-up in {} seconds \n", modelSize, getDurationInSeconds(loadStart));
            } else {
                throw e;
            }
        }

        lastSaveWasSuccessful.set(true);
    }

    protected static void addModelSaveSafeShutdownHook() {
        getRuntime().addShutdownHook(new Thread(() -> {
            LOG.warn("Shutdown initiated");
            while (savingInProgress.get()) {
                sleepSeconds(1);
            }
            LOG.warn("Exiting, nothing needs to be cleaned up");
        }));
    }
}