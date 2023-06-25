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

package org.tarik.train.qa.sequence.model.google.adapted;

import java.util.StringJoiner;

public class GoogleQuestionOnlyUnit {
    protected String question;
    protected String context;
    protected boolean relatesToQuestion;

    protected GoogleQuestionOnlyUnit() {
    }

    public GoogleQuestionOnlyUnit(String question, String context, boolean relatesToQuestion) {
        this.question = question;
        this.context = context;
        this.relatesToQuestion = relatesToQuestion;
    }

    public boolean isRelatedToQuestion() {
        return relatesToQuestion;
    }

    public String getQuestion() {
        return question + " ?";
    }

    public String getContext() {
        return context;
    }

    @Override
    public String toString() {
        return new StringJoiner(", ", GoogleQuestionOnlyUnit.class.getSimpleName() + "[", "]")
                .add("question='" + question + "'")
                .add("context='" + context + "'")
                .toString();
    }
}