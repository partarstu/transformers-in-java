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

package org.tarik.core.network.models.transformer;

import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tarik.core.data.IDataProvider;
import org.tarik.core.vocab.WordPieceVocab;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;

import static java.lang.Math.abs;
import static java.lang.String.format;
import static java.util.Map.Entry.comparingByValue;
import static java.util.Objects.requireNonNull;
import static java.util.function.Function.identity;
import static java.util.stream.Collectors.toCollection;
import static java.util.stream.Collectors.toMap;

/**
 * Generic extension of {@link AbstractTransformerSameDiffModel} with regard to processing the sequences. Contains the common functionality
 * like fetching new text sequences batch, providing prediction positions of the tokens which are least frequently used during the course
 * of the current training session.
 *
 * @param <T> the type of the child model
 */
public abstract class AbstractOpenDatasetTransformerModel<T extends AbstractOpenDatasetTransformerModel<T, U>,
        U extends AbstractOpenDatasetTransformerModel.PassageTokensSequence>
        extends AbstractTransformerSameDiffModel<T> {
    @Serial
    private static final long serialVersionUID = 6399581063535743259L;
    private static final Logger LOG = LoggerFactory.getLogger(AbstractOpenDatasetTransformerModel.class);
    protected static final String PUNCTUATIONS_PATTERN = "[.:,\"'_()\\[\\]|/;{}\\\\>\\-<*]+";
    protected transient BertWordPieceTokenizerFactory tokenizerFactory;
    protected transient IDataProvider dataProvider;
    protected final List<String> englishConjunctors;
    protected final int minimumSequenceUtilizationPercentage;
    protected final int totalSequenceCapacity;

    public AbstractOpenDatasetTransformerModel(int batchSize, int contextSequenceLength, int modelTestFrequency,
                                               int loggingFrequency, double learningRate, int hiddenSize, int layersAmount,
                                               int attentionHeadsAmount, int intermediateLayerSize, double labelSmoothing,
                                               boolean fixWeights, boolean fixTokenEmbeddings, int saveFrequencyInSteps, double beta2,
                                               float dropout, List<String> englishConjunctors, int minimumSequenceUtilizationPercentage,
                                               int totalSequenceCapacity) {
        super(batchSize, contextSequenceLength, modelTestFrequency, loggingFrequency, learningRate, hiddenSize, layersAmount,
                attentionHeadsAmount, intermediateLayerSize, labelSmoothing, fixWeights, fixTokenEmbeddings, saveFrequencyInSteps, beta2,
                dropout);
        this.englishConjunctors = englishConjunctors;
        this.minimumSequenceUtilizationPercentage = minimumSequenceUtilizationPercentage;
        this.totalSequenceCapacity = totalSequenceCapacity;
    }

    /**
     * Loads the model from the file to which it was previously saved using
     * {@link AbstractOpenDatasetTransformerModel#save(Path, boolean)} method. After loading the model,
     * {@link AbstractOpenDatasetTransformerModel#tokenizerFactory} is initialized.
     *
     * @param zipFilePath path to the ZIP file where the model file and the flat buffers file are stored
     * @param loadUpdater whether the stored during training updater's internal state should be loaded (is useful if the model was
     *                    saved during training and is being loaded to continue training where it's left)
     * @return the loaded model
     * @throws IOException in case any file-related issues arise
     */
    @Override
    public Optional<T> loadModelDataFromFile(Path zipFilePath, boolean loadUpdater) throws IOException {
        var loadedModel = super.loadModelDataFromFile(zipFilePath, loadUpdater);
        // vocab could be updated after loading the model, so the following step should occur after the load
        this.tokenizerFactory = new BertWordPieceTokenizerFactory(this.vocab.getTokenReader(), true, true, WordPieceVocab.DEFAULT_CHAR_SET);
        return loadedModel;
    }

    /**
     * Setter for the data provider.
     *
     * @param dataProvider concrete instance of {@link IDataProvider}
     */
    public void setDataProvider(IDataProvider dataProvider) {
        this.dataProvider = requireNonNull(dataProvider);
    }

    protected void addLogInfo(List<? extends PassageTokensSequence> batchedTokenSentences) {
        int amountOfTokens = batchedTokenSentences.stream()
                .mapToInt(tokenSequence -> tokenSequence.wordPieceTokens.size())
                .sum();
        double sequenceLengthUtilization = batchedTokenSentences.stream()
                .mapToInt(tokenSequence -> tokenSequence.wordPieceTokens.size())
                .average()
                .orElse(0) * 100 / baseSequenceLength;
        LOG.debug("Processing {} sequences with {} tokens with average sequence capacity utilization {}%",
                batchedTokenSentences.size(), amountOfTokens, format("%.1f", sequenceLengthUtilization));
    }

    protected abstract U generateTruncatedPassageTokensSequence(LinkedList<String> sentencesQueue);

    protected synchronized void fetchNewDataBlock(List<U> collector, Function<List<String>, Boolean> isLimitReachedFunction) {
        var sentencesQueue = new LinkedList<>(dataProvider.getPassages(isLimitReachedFunction));
        if (!sentencesQueue.isEmpty()) {
            while (!sentencesQueue.isEmpty()) {
                U passageTokensSequence = generateTruncatedPassageTokensSequence(sentencesQueue);
                if (passageTokensSequence.wordPieceTokens.size() >= baseSequenceLength * minimumSequenceUtilizationPercentage / 100) {
                    collector.add(passageTokensSequence);
                }
            }

            if (collector.size() < batchSize) {
                // Recursive collection till the required amount is reached
                fetchNewDataBlock(collector, isLimitReachedFunction);
            }
        }
    }

    protected List<U> getNewTokenSentencesBatch(LinkedList<U> collector, Function<List<String>, Boolean> isLimitReachedFunction) {
        List<U> batchedPassageTokenSequences = new LinkedList<>();
        // Fetching new data only if the collector is empty
        if (collector.isEmpty()) {
            fetchNewDataBlock(collector, isLimitReachedFunction);
        }
        while (batchedPassageTokenSequences.size() < batchSize && !collector.isEmpty()) {
            var tokensSequence = collector.pollFirst();
            batchedPassageTokenSequences.add(tokensSequence);
            if (collector.isEmpty()) {
                fetchNewDataBlock(collector, isLimitReachedFunction);
            }
        }
        return batchedPassageTokenSequences;
    }

    protected void addLeastFrequentlyPredictedTokenPositions(int expectedPredictionsAmount, Set<TokenUsageStats> collectedTokenUsageStats,
                                                             int distanceBetweenPredictions, Map<Integer, String> candidateTokensByPosition,
                                                             Map<Integer, String> notYetKnownTokensByPosition,
                                                             LinkedList<Integer> predictionPositions) {
        var tokenUsageStatsByTokenPosition = getTokenUsageStatsByTokenPosition(collectedTokenUsageStats, candidateTokensByPosition,
                notYetKnownTokensByPosition);
        int retries = 0;
        while (predictionPositions.size() < expectedPredictionsAmount && !tokenUsageStatsByTokenPosition.isEmpty() &&
                retries < candidateTokensByPosition.size()) {
            Pair<Integer, TokenUsageStats> nextCandidatePair = tokenUsageStatsByTokenPosition.pollFirst();
            int nextPositionCandidate = nextCandidatePair.getKey();
            if (predictionPositions.stream().noneMatch(predictionPosition ->
                    abs(predictionPosition - nextPositionCandidate) < distanceBetweenPredictions)) {
                predictionPositions.add(nextPositionCandidate);
                nextCandidatePair.getValue().incrementUsageFrequency();
            } else {
                ++retries;
                tokenUsageStatsByTokenPosition.addLast(nextCandidatePair);
            }
        }
    }

    @Nonnull
    protected LinkedList<Pair<Integer, TokenUsageStats>> getTokenUsageStatsByTokenPosition(Set<TokenUsageStats> collectedTokenUsageStats,
                                                                                           Map<Integer, String> candidateTokensByPosition,
                                                                                           Map<Integer, String> notYetKnownTokensByPosition) {
        Map<String, TokenUsageStats> relevantUsageStatsByToken = collectedTokenUsageStats.stream()
                .filter(tokenUsageStats -> candidateTokensByPosition.containsValue(tokenUsageStats.getToken()))
                .collect(toMap(TokenUsageStats::getToken, identity()));
        return candidateTokensByPosition.entrySet().stream()
                .filter(sequenceTokenByPosition -> !notYetKnownTokensByPosition
                        .containsKey(sequenceTokenByPosition.getKey()))
                .map(sequenceTokenByPosition -> Pair.of(sequenceTokenByPosition.getKey(),
                        relevantUsageStatsByToken.get(sequenceTokenByPosition.getValue())))
                // Placing the least used for prediction tokens first
                .sorted(comparingByValue())
                .collect(toCollection(LinkedList::new));
    }

    protected void addNotYetPredictedTokenPositions(int expectedPredictionsAmount, Set<TokenUsageStats> collectedTokenUsageStats,
                                                    int distanceBetweenPredictions, Map<Integer, String> candidateTokensByPosition,
                                                    Map<Integer, String> notYetKnownTokensByPosition,
                                                    LinkedList<Integer> predictionPositionsCollector) {
        LinkedList<Integer> firstPriorityPositions = new LinkedList<>(notYetKnownTokensByPosition.keySet());
        int retries = 0;
        while (predictionPositionsCollector.size() < expectedPredictionsAmount && !firstPriorityPositions.isEmpty() &&
                retries < candidateTokensByPosition.size()) {
            int nextPositionCandidate = firstPriorityPositions.pollFirst();
            if (predictionPositionsCollector.stream().noneMatch(predictionPosition ->
                    abs(predictionPosition - nextPositionCandidate) < distanceBetweenPredictions)) {
                predictionPositionsCollector.add(nextPositionCandidate);
                addTokenUsageStats(collectedTokenUsageStats, notYetKnownTokensByPosition.get(nextPositionCandidate));
            } else {
                ++retries;
                firstPriorityPositions.addLast(nextPositionCandidate);
            }
        }
    }

    protected void addTokenUsageStats(Set<TokenUsageStats> collectedTokenUsageStats, String token) {
        Optional<TokenUsageStats> existingStatsOptional = collectedTokenUsageStats.stream()
                .filter(tokenUsageStats -> tokenUsageStats.getToken().equals(token))
                .findAny();
        existingStatsOptional.ifPresentOrElse(TokenUsageStats::incrementUsageFrequency,
                () -> collectedTokenUsageStats.add(new TokenUsageStats(token)));
    }

    protected void decrementTokenUsageStats(Set<TokenUsageStats> collectedTokenUsageStats, String token) {
        collectedTokenUsageStats.stream()
                .filter(tokenUsageStats -> tokenUsageStats.getToken().equals(token))
                .forEach(TokenUsageStats::decrementUsageFrequency);
    }

    public static class PassageTokensSequence {
        protected List<String> wordPieceTokens;

        public PassageTokensSequence(List<String> wordPieceTokens) {
            this.wordPieceTokens = wordPieceTokens;
        }

        public List<String> getWordPieceTokens() {
            return wordPieceTokens;
        }
    }

    protected static class TokenUsageStats implements Comparable<TokenUsageStats>, Serializable {
        private final String token;
        private int usageFrequency;

        public TokenUsageStats(String token) {
            this.token = requireNonNull(token, "Token must be provided for stats colle");
            this.usageFrequency = 1;
        }

        public void incrementUsageFrequency() {
            ++this.usageFrequency;
        }

        public void decrementUsageFrequency() {
            --this.usageFrequency;
        }

        public String getToken() {
            return token;
        }

        @Override
        public String toString() {
            return new StringJoiner(", ", TokenUsageStats.class.getSimpleName() + "[", "]")
                    .add("token='" + token + "'")
                    .add("usageFrequency=" + usageFrequency)
                    .toString();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            TokenUsageStats that = (TokenUsageStats) o;
            return com.google.common.base.Objects.equal(this.token, that.token);
        }

        @Override
        public int hashCode() {
            return com.google.common.base.Objects.hashCode(token);
        }

        @Override
        public int compareTo(@Nonnull TokenUsageStats other) {
            return this.usageFrequency - other.usageFrequency;
        }
    }
}