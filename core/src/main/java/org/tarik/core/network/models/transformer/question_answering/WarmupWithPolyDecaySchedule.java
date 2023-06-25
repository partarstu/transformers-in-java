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

package org.tarik.core.network.models.transformer.question_answering;

import org.nd4j.linalg.schedule.ISchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serial;

import static com.google.common.base.Preconditions.checkArgument;
import static java.lang.Math.*;

/**
 * An experimental implementation of {@link ISchedule} based on polynomial decay function.
 */
public class WarmupWithPolyDecaySchedule implements ISchedule {
    @Serial
    private static final long serialVersionUID = -6774904472196103941L;
    private static final Logger LOG = LoggerFactory.getLogger(WarmupWithPolyDecaySchedule.class);
    protected int maxStepsAmount;
    protected int warmupStepsAmount;
    protected double initialLearningRate;
    private double power;
    protected int decayStepsAmount;
    protected int stepsProcessedAmount;
    protected double minimumLearningRate;
    private long lastIteration = -1;
    private double reducedLearningRate;

    private WarmupWithPolyDecaySchedule() {
        // Reflection
    }

    public WarmupWithPolyDecaySchedule(int maxStepsAmount, int warmupStepsAmount, double initialLearningRate,
                                       double minimumLearningRate, double power) {
        checkArgument(maxStepsAmount >= warmupStepsAmount,
                "Number of warmup steps (%s) can't exceed the total amount of steps provided (%s)",
                warmupStepsAmount, maxStepsAmount);
        this.maxStepsAmount = maxStepsAmount;
        this.warmupStepsAmount = warmupStepsAmount;
        this.initialLearningRate = initialLearningRate;
        this.minimumLearningRate = minimumLearningRate;
        this.power = power;
        this.decayStepsAmount = maxStepsAmount - warmupStepsAmount;
        LOG.info("""
                Initialized a new updater schedule with the following params:
                - learning rate {}
                - minimum learning rate {}
                - warmup steps {}
                - after-warmup steps {}
                - power {}
                """, initialLearningRate, minimumLearningRate, warmupStepsAmount, decayStepsAmount, power);
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        // This method can be called multiple times during the same iteration, incrementing steps should be done only once
        if (lastIteration != iteration) {
            stepsProcessedAmount++;
            lastIteration = iteration;

            if (stepsProcessedAmount >= warmupStepsAmount) {
                this.reducedLearningRate = stepsProcessedAmount > maxStepsAmount ? 0 :
                        initialLearningRate * pow(1 - ((stepsProcessedAmount - warmupStepsAmount) / (double) decayStepsAmount), power);
            } else {
                this.reducedLearningRate = ((stepsProcessedAmount + 1) / (double) warmupStepsAmount) * initialLearningRate;
            }
        }

        return max(minimumLearningRate, reducedLearningRate);
    }

    @Override
    public ISchedule clone() {
        return new WarmupWithPolyDecaySchedule(maxStepsAmount, warmupStepsAmount, initialLearningRate, power,
                stepsProcessedAmount);
    }

    public int getStepsProcessedAmount() {
        return stepsProcessedAmount;
    }

    public int getMaxStepsAmount() {
        return maxStepsAmount;
    }
}