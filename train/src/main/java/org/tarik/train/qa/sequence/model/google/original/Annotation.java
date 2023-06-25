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

import com.google.gson.annotations.SerializedName;

import java.util.List;

public class Annotation {
    @SerializedName("yes_no_answer")
    private String yesNo;

    @SerializedName("long_answer")
    private LongAnswerRange longAnswerRange;

    @SerializedName("short_answers")
    private List<AnswerRange> shortAnswerRanges;

    public String getYesNo() {
        return yesNo;
    }

    public LongAnswerRange getLongAnswerRange() {
        return longAnswerRange;
    }

    public List<AnswerRange> getShortAnswers() {
        return shortAnswerRanges;
    }
}