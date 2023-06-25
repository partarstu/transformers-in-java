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

import org.apache.commons.lang.StringUtils;
import org.tarik.train.qa.sequence.model.google.adapted.GoogleMultipleContextsQaUnit;
import org.tarik.train.qa.sequence.model.google.adapted.GoogleQuestionAnswerUnit;
import org.tarik.train.qa.sequence.model.google.adapted.GoogleQuestionOnlyUnit;
import org.tarik.train.qa.sequence.model.google.original.Annotation;
import org.tarik.train.qa.sequence.model.google.original.AnswerRange;
import org.tarik.train.qa.sequence.model.google.original.QaUnit;

import java.util.*;

import static java.lang.String.join;
import static java.util.Arrays.copyOfRange;
import static java.util.Optional.empty;
import static java.util.Optional.ofNullable;
import static java.util.stream.Collectors.toSet;
import static org.tarik.utils.CommonUtils.decodeAsHtmlText;

public class QuestionAnswerUnitAdaptor {
    private static final Random random = new Random();

    public static GoogleQuestionAnswerUnit convertOriginalQaUnit(QaUnit originalUnit) {
        var defaultAnnotation = originalUnit.getAnnotations().get(0);
        String[] documentTokens = originalUnit.getDocumentText().split(" ");
        var longAnswerOptional =
                extractRangeAsHtmlDecodedString(defaultAnnotation.getLongAnswerRange(), documentTokens);
        String question = originalUnit.getQuestion();
        String context = longAnswerOptional.orElseGet(() -> getRandomContextCandidate(documentTokens, originalUnit));
        Optional<AnswerRange> availableAnswerRange = getFirstAvailableShortAnswerRange(defaultAnnotation);
        // Start position should be 0-based, the original one is 1-based
        String answer = availableAnswerRange.flatMap(answerRange ->
                        extractRangeAsHtmlDecodedString(answerRange, documentTokens))
                .orElse(null);
        return new GoogleQuestionAnswerUnit(question, answer, context);
    }

    public static Set<GoogleQuestionOnlyUnit> convertOriginalQaUnit(QaUnit originalUnit, int negativeSamplesAmount) {
        var defaultAnnotation = originalUnit.getAnnotations().get(0);
        String[] documentTokens = originalUnit.getDocumentText().split(" ");
        Set<GoogleQuestionOnlyUnit> questionOnlyUnits = new HashSet<>();
        String question = originalUnit.getQuestion();
        var longAnswerOptional =
                extractRangeAsHtmlDecodedString(defaultAnnotation.getLongAnswerRange(), documentTokens);
        longAnswerOptional.ifPresent(longAnswer ->
                questionOnlyUnits.add(new GoogleQuestionOnlyUnit(question, longAnswer, true)));

        getRandomContextCandidates(documentTokens, originalUnit, defaultAnnotation.getLongAnswerRange(), negativeSamplesAmount, 0)
                .stream()
                .map(context -> new GoogleQuestionOnlyUnit(question, context, false))
                .forEach(questionOnlyUnits::add);

        return questionOnlyUnits;
    }

    public static GoogleMultipleContextsQaUnit convertIntoMultipleContextsUnit(QaUnit originalUnit, int amountOfContextsPerUnit,
                                                                               int minAmountOfTokensInContext) {
        var defaultAnnotation = originalUnit.getAnnotations().get(0);
        String[] documentTokens = originalUnit.getDocumentText().split(" ");
        Optional<AnswerRange> availableAnswerRange = getFirstAvailableShortAnswerRange(defaultAnnotation);
        String answer = availableAnswerRange.flatMap(answerRange -> extractRangeAsHtmlDecodedString(answerRange, documentTokens))
                .orElse(null);
        String question = originalUnit.getQuestion();
        Set<String> contexts = new HashSet<>();
        var longAnswerOptional = extractRangeAsHtmlDecodedString(defaultAnnotation.getLongAnswerRange(), documentTokens);
        longAnswerOptional.ifPresent(contexts::add);

        contexts.addAll(getRandomContextCandidates(documentTokens, originalUnit, defaultAnnotation.getLongAnswerRange(),
                amountOfContextsPerUnit - contexts.size(), minAmountOfTokensInContext));
        return new GoogleMultipleContextsQaUnit(question, answer, contexts);

    }

    private static String getRandomContextCandidate(String[] tokens, QaUnit originalUnit) {
        var longAnswerCandidates = originalUnit.getLongAnswerCandidates();
        return random.ints(0, longAnswerCandidates.size())
                .boxed()
                .map(longAnswerCandidates::get)
                .findFirst()
                .flatMap(candidateRange -> extractRangeAsHtmlDecodedString(candidateRange, tokens)).orElseThrow();
    }

    private static Set<String> getRandomContextCandidates(String[] tokens, QaUnit originalUnit,
                                                          AnswerRange answerRangeToExclude, int amount, int minAmountOfTokensInContext) {
        var acceptableContexts = originalUnit.getLongAnswerCandidates().stream()
                .filter(longAnswerCandidate -> !longAnswerCandidate.equals(answerRangeToExclude))
                .filter(longAnswerCandidate -> longAnswerCandidate.getEndToken() - longAnswerCandidate.getStartToken() >
                        minAmountOfTokensInContext)
                .flatMap(candidateRange -> extractRangeAsHtmlDecodedString(candidateRange, tokens).stream())
                .filter(StringUtils::isNotBlank)
                .distinct()
                .toList();
        if (acceptableContexts.size() > amount) {
            return random.ints(0, acceptableContexts.size())
                    .distinct()
                    .boxed()
                    .map(acceptableContexts::get)
                    .limit(amount)
                    .collect(toSet());
        } else {
            return Set.copyOf(acceptableContexts);
        }
    }

    private static Optional<AnswerRange> getFirstAvailableShortAnswerRange(Annotation defaultAnnotation) {
        var contextRange = defaultAnnotation.getLongAnswerRange();
        return ofNullable(defaultAnnotation.getShortAnswers())
                .orElse(List.of())
                .stream()
                .filter(range -> range.getStartToken() >= contextRange.getStartToken() &&
                        range.getEndToken() <= contextRange.getEndToken())
                .findFirst();
    }

    private static Optional<String> extractRangeAsHtmlDecodedString(AnswerRange range, String[] tokens) {
        int startToken = range.getStartToken();
        int endToken = range.getEndToken();
        if (startToken >= 0 && endToken >= 0 && endToken > startToken) {
            return decodeAsHtmlText(join(" ", copyOfRange(tokens, startToken, endToken)));
        } else {
            return empty();
        }
    }
}