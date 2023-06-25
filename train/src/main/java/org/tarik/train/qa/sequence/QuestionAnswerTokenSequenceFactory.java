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

import com.google.common.collect.ImmutableList;
import org.apache.commons.collections4.CollectionUtils;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tarik.core.network.models.transformer.question_answering.sequence.IQaTokenSequence;
import org.tarik.train.qa.sequence.model.google.adapted.GoogleMultipleContextsQaUnit;
import org.tarik.train.qa.sequence.model.google.adapted.GoogleQuestionAnswerUnit;
import org.tarik.train.qa.sequence.model.google.adapted.GoogleQuestionOnlyUnit;
import org.tarik.train.qa.sequence.model.squad.Answer;
import org.tarik.train.qa.sequence.model.squad.QaData;
import org.tarik.train.qa.sequence.model.squad.QuestionAnswerBlock;
import org.tarik.train.qa.sequence.model.squad.Squad;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Stream;

import static java.lang.String.valueOf;
import static java.util.Collections.indexOfSubList;
import static java.util.Collections.lastIndexOfSubList;
import static java.util.Objects.requireNonNull;
import static java.util.Optional.*;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static org.apache.commons.lang3.StringUtils.isNotBlank;
import static org.nd4j.common.base.Preconditions.checkArgument;

public class QuestionAnswerTokenSequenceFactory {
    private static final String NON_LETTER_REGEX = "\\P{L}";
    private static final Logger LOG = LoggerFactory.getLogger(QuestionAnswerTokenSequenceFactory.class);

    public static List<QuestionAnswerTokenSequence> fromSquad(
            Squad squad, BertWordPieceTokenizerFactory tokenizerFactory, int modelSequenceLengthWithoutSeparators) {
        Function<QuestionAnswerTokenSequence, Integer> positionShiftProvider = qaTokenSequence ->
                modelSequenceLengthWithoutSeparators - qaTokenSequence.getQuestionTokens().size();
        List<QuestionAnswerTokenSequence> qaTokenSequences = squad.getData().stream()
                .map(QaData::getParagraphs)
                .flatMap(Collection::stream)
                .map(par -> par.getQas().stream()
                        .map(qaBlock -> fromQaBlock(qaBlock, par.getContext(), tokenizerFactory))
                        .flatMap(Collection::stream)
                )
                .flatMap(Stream::distinct)
                .filter(qaTokenSequence ->
                        qaTokenSequence.getAnswerTokenStartPosition() < positionShiftProvider.apply(qaTokenSequence))
                .collect(toList());

        // Context must be truncated in case it's longer than the configured model's sequence length
        truncateContextIfNeeded(qaTokenSequences, positionShiftProvider);

        // It might happen that the answer is located in the part of the context which is beyond the configured
        // sequence length. In this case answerTokenEndPosition must be adapted to fit the config
        alterAnswerTokenEndPositionIfNeeded(qaTokenSequences, positionShiftProvider);
        return qaTokenSequences;
    }

    public static List<QuestionAnswerTokenSequence> fromGoogleQaUnits(
            Collection<GoogleQuestionAnswerUnit> qaUnits, BertWordPieceTokenizerFactory tokenizerFactory,
            int modelSequenceLengthWithoutSeparators) {
        Function<QuestionAnswerTokenSequence, Integer> positionShiftProvider = qaTokenSequence ->
                modelSequenceLengthWithoutSeparators - qaTokenSequence.getQuestionTokens().size();
        var qaTokenSequences = getTruncatedQaTokenSequences(qaUnits,
                modelSequenceLengthWithoutSeparators, unit -> fromGoogleQaUnit(unit, tokenizerFactory));

        // It might happen that the answer is located in the part of the context which is beyond the configured
        // sequence length. In this case answerTokenEndPosition must be adapted to fit the config
        alterAnswerTokenEndPositionIfNeeded(qaTokenSequences, positionShiftProvider);
        return qaTokenSequences;
    }

    public static Optional<QuestionAnswerTokenSequence> newOnesFromGoogleQaUnit(
            GoogleQuestionAnswerUnit qaUnit, BertWordPieceTokenizerFactory tokenizerFactory,
            int modelSequenceLengthWithoutSeparators) {
        Function<QuestionAnswerTokenSequence, Integer> positionShiftProvider = qaTokenSequence ->
                modelSequenceLengthWithoutSeparators;
        List<QuestionAnswerTokenSequence> qaTokenSequences = Stream.of(qaUnit)
                .map(unit -> fromGoogleQaUnit(unit, tokenizerFactory))
                .flatMap(Optional::stream)
                .filter(qaTokenSequence ->
                        qaTokenSequence.getAnswerTokenStartPosition() < modelSequenceLengthWithoutSeparators)
                .collect(toList());

        // It might happen that the answer is located in the part of the context which is beyond the configured
        // sequence length. In this case answerTokenEndPosition must be adapted to fit the config
        alterAnswerTokenEndPositionIfNeeded(qaTokenSequences, positionShiftProvider);
        return qaTokenSequences.stream().findAny();
    }

    public static List<IQaTokenSequence> fromGoogleMultiContextQaUnits(
            Collection<GoogleMultipleContextsQaUnit> qaUnits, BertWordPieceTokenizerFactory tokenizerFactory,
            int modelSequenceLengthWithoutSeparators) {
        return getTruncatedQaTokenSequences(qaUnits, tokenizerFactory, modelSequenceLengthWithoutSeparators);
    }

    public static List<QuestionAnswerTokenSequence> fromGoogleQuestionUnits(
            Collection<GoogleQuestionOnlyUnit> questionOnlyUnits, BertWordPieceTokenizerFactory tokenizerFactory,
            int modelSequenceLengthWithoutSeparators) {
        return getTruncatedQaTokenSequences(questionOnlyUnits, modelSequenceLengthWithoutSeparators,
                unit -> fromGoogleQuestionUnit(unit, tokenizerFactory));
    }

    public static QuestionAnswerTokenSequence fromGoogleQuestionUnit(
            GoogleQuestionOnlyUnit questionOnlyUnit, BertWordPieceTokenizerFactory tokenizerFactory,
            int modelSequenceLengthWithoutSeparators) {
        return getTruncatedQaTokenSequences(List.of(questionOnlyUnit), modelSequenceLengthWithoutSeparators,
                unit -> fromGoogleQuestionUnit(unit, tokenizerFactory)).get(0);
    }

    public static IQaTokenSequence fromQuestionAndMultipleContexts(String question, Collection<String> contexts,
                                                                   BertWordPieceTokenizerFactory tokenizerFactory,
                                                                   int modelSequenceLengthWithoutSeparators) {
        Function<IQaTokenSequence, Integer> positionShiftProvider = qaTokenSequence ->
                modelSequenceLengthWithoutSeparators - qaTokenSequence.getQuestionTokens().length;
        List<String> questionTokens = tokenizerFactory.create(question).getTokens();
        List<String[]> contextTokensList = contexts.stream()
                .map(tokenizerFactory::create)
                .map(Tokenizer::getTokens)
                .filter(CollectionUtils::isNotEmpty)
                .map(tokens -> tokens.toArray(String[]::new))
                .collect(toList());
        var multipleContextsQaTokenSequence = new MultipleContextsQaTokenSequence(questionTokens, contextTokensList, List.of());
        truncateContextsIfNeeded(List.of(multipleContextsQaTokenSequence), positionShiftProvider);

        return multipleContextsQaTokenSequence;
    }

    private static List<IQaTokenSequence> getTruncatedQaTokenSequences(
            Collection<GoogleMultipleContextsQaUnit> originalUnits, BertWordPieceTokenizerFactory tokenizerFactory,
            int modelSequenceLengthWithoutSeparators) {
        Function<IQaTokenSequence, Integer> positionShiftProvider = qaTokenSequence ->
                modelSequenceLengthWithoutSeparators - qaTokenSequence.getQuestionTokens().length;
        List<IQaTokenSequence> qaTokenSequences = originalUnits.parallelStream()
                .filter(GoogleMultipleContextsQaUnit::hasAnswer)
                .map(multipleContextsQaUnit -> fromGoogleMultipleContextsQaUnit(multipleContextsQaUnit, tokenizerFactory))
                .flatMap(Optional::stream)
                .collect(toList());

        // Context must be truncated in case it's longer than the configured model's sequence length
        truncateContextsIfNeeded(qaTokenSequences, positionShiftProvider);
        return qaTokenSequences;
    }

    private static <T> List<QuestionAnswerTokenSequence> getTruncatedQaTokenSequences(
            Collection<T> originalUnits, int modelSequenceLengthWithoutSeparators,
            Function<T, Optional<QuestionAnswerTokenSequence>> mapper) {
        Function<QuestionAnswerTokenSequence, Integer> positionShiftProvider = qaTokenSequence ->
                modelSequenceLengthWithoutSeparators - qaTokenSequence.getQuestionTokens().size();
        List<QuestionAnswerTokenSequence> qaTokenSequences = originalUnits.stream()
                .map(mapper)
                .flatMap(Optional::stream)
                .filter(qaTokenSequence ->
                        qaTokenSequence.getAnswerTokenStartPosition() < positionShiftProvider.apply(qaTokenSequence))
                .collect(toList());

        // Context must be truncated in case it's longer than the configured model's sequence length
        truncateContextIfNeeded(qaTokenSequences, positionShiftProvider);
        return qaTokenSequences;
    }

    public static List<QuestionAnswerTokenSequence> fromQuestionAndContexts(
            String question, Collection<String> contexts, BertWordPieceTokenizerFactory tokenizerFactory,
            int modelSequenceLengthWithoutSeparators) {
        Function<QuestionAnswerTokenSequence, Integer> positionShiftProvider = qaTokenSequence ->
                modelSequenceLengthWithoutSeparators - qaTokenSequence.getQuestionTokens().size();
        List<QuestionAnswerTokenSequence> questionAnswerTokenSequences = contexts.stream()
                .map(context -> withNoAnswerTokens(question, context, tokenizerFactory))
                .collect(toList());
        truncateContextIfNeeded(questionAnswerTokenSequences, positionShiftProvider);
        return questionAnswerTokenSequences;
    }

    private static QuestionAnswerTokenSequence withNoAnswerTokens(String question, String context,
                                                                  BertWordPieceTokenizerFactory tokenizerFactory) {
        List<String> questionTokens = tokenizerFactory.create(question).getTokens();
        List<String> contextTokens = tokenizerFactory.create(context).getTokens();

        return noAnswerTokensQaTokenSequence(questionTokens, contextTokens);
    }

    private static Optional<IQaTokenSequence> fromGoogleMultipleContextsQaUnit(
            GoogleMultipleContextsQaUnit multipleContextsQaUnit,
            BertWordPieceTokenizerFactory tokenizerFactory) {
        List<String> questionTokens = tokenizerFactory.create(multipleContextsQaUnit.getQuestion()).getTokens();
        List<String[]> contextTokensList = multipleContextsQaUnit.getContexts().stream()
                .map(tokenizerFactory::create)
                .map(Tokenizer::getTokens)
                .filter(CollectionUtils::isNotEmpty)
                .map(tokens -> tokens.toArray(String[]::new))
                .collect(toList());
        List<String> answerTokens = tokenizerFactory.create(multipleContextsQaUnit.getAnswer()).getTokens();
        return ofNullable(answerTokens)
                .filter(CollectionUtils::isNotEmpty)
                .map(answer -> new MultipleContextsQaTokenSequence(questionTokens, contextTokensList, answer));
    }

    private static List<QuestionAnswerTokenSequence> fromQaBlock(QuestionAnswerBlock questionAnswerBlock,
                                                                 String context,
                                                                 BertWordPieceTokenizerFactory tokenizerFactory) {
        if (questionAnswerBlock.isImpossible()) {
            List<String> questionTokens = tokenizerFactory.create(questionAnswerBlock.getQuestion()).getTokens();
            List<String> contextTokens = tokenizerFactory.create(context).getTokens();
            return ImmutableList.of(new QuestionAnswerTokenSequence(questionTokens, contextTokens, -1, -1, false));
        } else {
            return questionAnswerBlock.getAnswers().stream()
                    .filter(Objects::nonNull)
                    .filter(answer -> isNotBlank(answer.getText()))
                    .map(answer -> fromExistingQa(questionAnswerBlock.getQuestion(), context, answer, tokenizerFactory))
                    .collect(toList());
        }
    }

    private static Optional<QuestionAnswerTokenSequence> fromGoogleQaUnit(GoogleQuestionAnswerUnit qaUnit,
                                                                          BertWordPieceTokenizerFactory tokenizerFactory) {
        List<String> questionTokens = tokenizerFactory.create(qaUnit.getQuestion()).getTokens();
        List<String> contextTokens = tokenizerFactory.create(qaUnit.getContext()).getTokens();
        if (!qaUnit.hasAnswer()) {
            return of(noAnswerTokensQaTokenSequence(questionTokens, contextTokens));
        }
        List<String> answerTokens = tokenizerFactory.create(qaUnit.getAnswer()).getTokens();
        if (answerTokens.isEmpty()) {
            return of(noAnswerTokensQaTokenSequence(questionTokens, contextTokens));
        } else {
            int actualAnswerFirstOccurrence = indexOfSubList(contextTokens, answerTokens);
            int actualAnswerLastOccurrence = lastIndexOfSubList(contextTokens, answerTokens);
            if (actualAnswerFirstOccurrence < 0) {
                return empty();
            }
            if (actualAnswerFirstOccurrence != actualAnswerLastOccurrence) {
                // Can't handle multiple answer instances inside the same context
                return empty();
            } else {
                return of(new QuestionAnswerTokenSequence(questionTokens, contextTokens, actualAnswerFirstOccurrence,
                        actualAnswerFirstOccurrence + answerTokens.size(), true));
            }
        }
    }

    private static Optional<QuestionAnswerTokenSequence> fromGoogleQuestionUnit(GoogleQuestionOnlyUnit qaUnit,
                                                                                BertWordPieceTokenizerFactory tokenizerFactory) {
        List<String> questionTokens = tokenizerFactory.create(qaUnit.getQuestion()).getTokens();
        List<String> contextTokens = tokenizerFactory.create(qaUnit.getContext()).getTokens();

        return of(new QuestionAnswerTokenSequence(questionTokens, contextTokens, -1, -1, qaUnit.isRelatedToQuestion()));
    }

    private static QuestionAnswerTokenSequence noAnswerTokensQaTokenSequence(List<String> questionTokens,
                                                                             List<String> contextTokens) {
        return new QuestionAnswerTokenSequence(questionTokens, contextTokens, -1, -1, false);
    }

    private static QuestionAnswerTokenSequence fromExistingQa(String question, String context, Answer answer,
                                                              BertWordPieceTokenizerFactory tokenizerFactory) {
        requireNonNull(answer);
        checkArgument(isNotBlank(answer.getText()));
        List<String> questionTokens = tokenizerFactory.create(question).getTokens();
        List<String> contextTokens = tokenizerFactory.create(context).getTokens();

        int answerStartPosition = answer.getAnswerStart();
        String answerText = answer.getText();
        String[] answerWords = answerText.split("\\s");

        // It might happen that the answer starts not with a full word but part of it. In this case the
        // answer first token position must be calculated based on the full word tokens
        int answerTokenStartPosition;
        int answerTokenEndPosition;
        List<String> answerTokens = new LinkedList<>(tokenizerFactory.create(answerText).getTokens());

        if (answerStartPosition == 0 ||
                (valueOf(context.charAt(answerStartPosition - 1)).matches(NON_LETTER_REGEX))) {
            List<String> contextTokensBeforeAnswer =
                    tokenizerFactory.create(context.substring(0, answerStartPosition)).getTokens();
            answerTokenStartPosition = contextTokensBeforeAnswer.size();
        } else {
            int answerFirstFullWordPosition = context.substring(0, answerStartPosition)
                    .lastIndexOf(" ") + 1;
            int answerFirstFullWordEndPosition = context.indexOf(" ", answerFirstFullWordPosition);
            if (answerFirstFullWordEndPosition == -1) {
                answerFirstFullWordEndPosition = context.length();
            }
            String firstFullWord = context.substring(answerFirstFullWordPosition, answerFirstFullWordEndPosition);
            List<String> answerFirstFullWordTokens = tokenizerFactory.create(firstFullWord).getTokens();
            try {

                String answerFirstWord = answerWords[0];
                String answerFirstWordToken = tokenizerFactory.create(answerFirstWord).getTokens().get(0);
                if (answerFirstFullWordTokens.contains(answerFirstWord)) {
                    // A case when the answer's first word is not a standalone word but it's a single token
                    answerTokenStartPosition = answerFirstFullWordTokens.indexOf(answerFirstWord);
                } else if (answerFirstFullWordTokens.contains(answerFirstWordToken)) {
                    // A case when the answer's first word is not a standalone word but it's more than a single token
                    answerTokenStartPosition = answerFirstFullWordTokens.indexOf(answerFirstWordToken);
                } else {
                    // A case when the answer's first word is neither a standalone word nor a standalone full word's
                    // token. It happens when the answer's first word tokens are not a sublist of the full word's tokens
                    // For this case it's necessary to find the index of the full word's token which contains the
                    // answer's first word's first token, otherwise the beginning of the whole word will be used
                    answerTokenStartPosition = range(0, answerFirstFullWordTokens.size())
                            .filter(index -> answerFirstFullWordTokens.get(index).contains(answerFirstWordToken))
                            .findFirst()
                            .orElse(0);
                }
                // Answer tokens need to be corrected in order to contain the real answer first word's tokens

                answerTokens = new LinkedList<>(answerFirstFullWordTokens.subList(answerTokenStartPosition,
                        answerFirstFullWordTokens.size()));
                if (answerWords.length > 1) {
                    answerTokens.addAll(tokenizerFactory.create(
                                    answerText.substring(answerText.indexOf(answerWords[1])))
                            .getTokens());
                }

            } catch (Exception e) {
                LOG.warn("Got exception for question [{}], context [{}], answer [{}], answerFirstFullWordTokens " +
                                "[{}], answerFirstWordToken [{}]", question, context, answer,
                        tokenizerFactory.create(firstFullWord).getTokens(),
                        tokenizerFactory.create(answerWords[0]).getTokens().get(0));
                throw e;
            }
        }

        int answerSpanEndPosition = answerStartPosition + answerText.length();
        if (answerSpanEndPosition >= context.length() - 1) {
            answerTokenEndPosition = contextTokens.size() - 1;
        } else {
            answerTokenEndPosition = answerTokens.size() + answerTokenStartPosition;
        }

        return new QuestionAnswerTokenSequence(questionTokens, contextTokens, answerTokenStartPosition,
                answerTokenEndPosition, true);
    }


    private static void alterAnswerTokenEndPositionIfNeeded(
            Collection<QuestionAnswerTokenSequence> qaTokenSequencesStream,
            Function<QuestionAnswerTokenSequence, Integer> positionShiftProvider) {
        qaTokenSequencesStream.stream()
                .filter(qaTokenSequence ->
                        qaTokenSequence.getAnswerTokenEndPosition() >= positionShiftProvider.apply(qaTokenSequence))
                .forEach(qaTokenSequence ->
                        qaTokenSequence
                                .setAnswerTokenEndPosition(positionShiftProvider.apply(qaTokenSequence) - 1));
    }

    private static void truncateContextIfNeeded(Collection<QuestionAnswerTokenSequence> qaTokenSequences,
                                                Function<QuestionAnswerTokenSequence, Integer> positionShiftProvider) {
        qaTokenSequences.stream()
                .filter(qaTokenSequence ->
                        qaTokenSequence.getContextTokens().size() >= positionShiftProvider.apply(qaTokenSequence))
                .forEach(qaTokenSequence ->
                        qaTokenSequence.truncateContext(positionShiftProvider.apply(qaTokenSequence)));
    }

    private static void truncateContextsIfNeeded(Collection<IQaTokenSequence> qaTokenSequences,
                                                 Function<IQaTokenSequence, Integer> positionShiftProvider) {
        qaTokenSequences.parallelStream().forEach(qaTokenSequence ->
                qaTokenSequence.truncateContexts(positionShiftProvider.apply(qaTokenSequence)));
    }
}