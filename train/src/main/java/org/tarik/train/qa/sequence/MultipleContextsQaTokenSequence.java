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
import com.google.common.base.Preconditions;
import org.tarik.core.network.models.transformer.question_answering.sequence.IQaTokenSequence;

import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import static java.util.Collections.shuffle;
import static org.nd4j.common.io.CollectionUtils.isEmpty;

public class MultipleContextsQaTokenSequence implements IQaTokenSequence {
    private static long idCounter;
    private final String[] questionTokens;
    private final List<String[]> contextTokensList;
    private final String[] answerTokens;
    private final long id;

    public MultipleContextsQaTokenSequence(List<String> questionTokens, Collection<String[]> contextTokensList,
                                           List<String> answerTokens) {
        Preconditions.checkArgument(!isEmpty(questionTokens), "MultipleContextsQaTokenSequence must have a question");
        Preconditions.checkArgument(!isEmpty(contextTokensList), "MultipleContextsQaTokenSequence must have at least 1 context");
        questionTokens.remove("?");
        this.questionTokens = questionTokens.toArray(String[]::new);
        this.contextTokensList = new LinkedList<>(contextTokensList);
        this.answerTokens = answerTokens.toArray(String[]::new);
        this.id = idCounter++;
    }

    @Override
    public String[] getQuestionTokens() {
        return questionTokens;
    }

    @Override
    public List<String[]> getShuffledContextTokensList() {
        var shuffledList = new LinkedList<>(contextTokensList) ;
        shuffle(shuffledList);
        return shuffledList;
    }

    @Override
    public String[] getAnswerTokens() {
        return answerTokens;
    }

    @Override
    public void truncateContexts(int newSize) {
        var contextsToTruncate = contextTokensList.stream()
                .filter(contextTokens -> newSize < contextTokens.length)
                .toList();
        contextsToTruncate.forEach(contextTokensList::remove);
        contextsToTruncate.forEach(context-> contextTokensList.add(Arrays.copyOf(context, newSize)));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        MultipleContextsQaTokenSequence that = (MultipleContextsQaTokenSequence) o;
        return id == that.id && Objects.equal(questionTokens, that.questionTokens) &&
                Objects.equal(contextTokensList, that.contextTokensList) &&
                Objects.equal(answerTokens, that.answerTokens);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(questionTokens, contextTokensList, answerTokens, id);
    }
}