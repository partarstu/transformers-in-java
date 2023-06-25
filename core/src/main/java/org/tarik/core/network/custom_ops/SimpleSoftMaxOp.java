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

package org.tarik.core.network.custom_ops;

import com.google.common.collect.ImmutableList;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;

import java.util.List;

/**
 * <p>
 * Experimental custom Operation which overrides the standard differentiation logic by simplifying it in case this OP is used for the
 * purposes of calculating the cross-entropy loss. In this case the differentiation logic is to simply deduct the labels from the logits
 * provided by the Softmax Op.
 * </p>
 * <p>
 * This Op needs to be explicitly added to the Samediff graph by using the mechanism of the User-Defined Functions. This part is till WIP
 * </p>
 */
public class SimpleSoftMaxOp extends SoftMax {
    protected SDVariable labels;

    public SimpleSoftMaxOp(SameDiff sameDiff, SDVariable input, int dimension, SDVariable labels) {
        super(sameDiff, new SDVariable[]{input}, dimension);
        this.labels = labels;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable deltas = outputVariable().sub(labels);
        return ImmutableList.of(deltas);
    }
}