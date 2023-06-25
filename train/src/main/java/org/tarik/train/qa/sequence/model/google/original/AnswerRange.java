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

package org.tarik.train.qa.sequence.model.google.original;

import com.google.common.base.Objects;
import com.google.gson.annotations.SerializedName;

public class AnswerRange {
    @SerializedName("start_token")
    protected int startToken;

    @SerializedName("end_token")
    protected int endToken;

    public int getStartToken() {
        return startToken;
    }

    public int getEndToken() {
        return endToken;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || !getClass().isAssignableFrom(o.getClass())) {
            return false;
        }
        AnswerRange that = (AnswerRange) o;
        return startToken == that.startToken && endToken == that.endToken;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(startToken, endToken);
    }
}