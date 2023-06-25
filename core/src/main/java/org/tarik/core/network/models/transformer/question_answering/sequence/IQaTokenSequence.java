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

package org.tarik.core.network.models.transformer.question_answering.sequence;

import java.util.List;

/**
 * An interface representing the behavior of a sequence related to the Question-Answering task and used by
 * {@link org.tarik.core.network.models.transformer.question_answering.GenerativeQuestionAnsweringModel}
 */
public interface IQaTokenSequence {
    /**
     * Provides the tokens of the question for the current question-answer sequence.
     *
     * @return question tokens array
     */
    String[] getQuestionTokens();

    /**
     * Provides the array of question-answer sequence context tokens. One question could have multiple contexts, related to it. That's
     * why the result is expected to be the list of token arrays, not a single tokens array. In order to provide no specific dependency
     * on the order of the contexts, they need to be shuffled.
     *
     * @return list of shuffled context token arrays
     */
    List<String[]> getShuffledContextTokensList();

    /**
     * Provides the tokens of the answer for the current question-answer sequence.
     *
     * @return answer tokens array
     */
    String[] getAnswerTokens();

    /**
     * Truncates each of the contexts stored in the current question-answer sequence in order to adhere to the specified size.
     *
     * @param newSize the maximum amount of tokens which could be present in each context
     */
    void truncateContexts(int newSize);
}