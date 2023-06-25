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

package org.tarik.train.qa.sequence;

import com.google.common.base.Objects;

import java.util.List;
import java.util.Optional;

import static java.util.Objects.requireNonNull;
import static java.util.Optional.empty;
import static java.util.Optional.of;

public class QuestionAnswerTokenSequence {
    private static long idCounter;
    private final List<String> questionTokens;
    private List<String> contextTokens;
    private final int answerTokenStartPosition;
    private int answerTokenEndPosition;
    private final boolean answerExists;
    private final long id;

    public QuestionAnswerTokenSequence(List<String> questionTokens, List<String> contextTokens, int answerTokenStartPosition,
                                       int answerTokenEndPosition, boolean answerExists) {

        this.questionTokens = requireNonNull(questionTokens);
        removeQuestionSign();
        this.contextTokens = contextTokens;
        this.answerTokenStartPosition = answerTokenStartPosition;
        this.answerTokenEndPosition = answerTokenEndPosition;
        this.answerExists = answerExists;
        this.id = idCounter++;
    }

    public List<String> getQuestionTokens() {
        return questionTokens;
    }

    public void removeQuestionSign(){
        this.questionTokens.remove("?");
    }

    public Optional<List<String>> getAnswerTokens(){
        if(answerExists){
            return of(contextTokens.subList(answerTokenStartPosition, answerTokenEndPosition));
        } else {
            return empty();
        }
    }

    public List<String> getContextTokens() {
        return contextTokens;
    }

    public int getAnswerTokenStartPosition() {
        return answerTokenStartPosition;
    }

    public int getAnswerTokenEndPosition() {
        return answerTokenEndPosition;
    }

    public void setAnswerTokenEndPosition(int answerTokenEndPosition) {
        this.answerTokenEndPosition = answerTokenEndPosition;
    }

    public boolean answerExists() {
        return answerExists;
    }

    public long getId() {
        return id;
    }

    public void truncateContext(int newSize) {
        if (newSize < this.contextTokens.size()) {
            this.contextTokens = this.contextTokens.subList(0, newSize);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        QuestionAnswerTokenSequence that = (QuestionAnswerTokenSequence) o;
        return answerTokenStartPosition == that.answerTokenStartPosition &&
                answerTokenEndPosition == that.answerTokenEndPosition && answerExists == that.answerExists &&
                id == that.id && Objects.equal(questionTokens, that.questionTokens) &&
                Objects.equal(contextTokens, that.contextTokens);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(questionTokens, contextTokens, answerTokenStartPosition, answerTokenEndPosition, answerExists, id);
    }
}