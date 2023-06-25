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

package org.tarik.core.vocab;

import org.apache.commons.io.input.CharSequenceInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.io.InputStream;
import java.io.Serial;
import java.io.Serializable;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;

import static java.lang.String.join;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Optional.ofNullable;
import static org.tarik.core.vocab.WordPieceVocab.WordPiece;
import static org.tarik.utils.CommonUtils.readLines;

/**
 * Represents a set as a vocabulary of {@link WordPiece} tokens. Each token has a look-up based on its string value its index. The
 * latter is used by the model during training.
 */
public class WordPieceVocab extends HashSet<WordPiece> implements Serializable {
    private static final Logger LOG = LoggerFactory.getLogger(WordPieceVocab.class);
    public static final Charset DEFAULT_CHAR_SET = UTF_8;
    public static final String WORD_CONJUNCTION_SYMBOL = "##";
    public static final String MASK_SYMBOL = "[MASK]";
    public static final String PADDING_SYMBOL = "[PAD]";
    public static final String CLASSIFICATION_SYMBOL = "[CLS]";
    public static final String END_OF_SENTENCE_SYMBOL = "[EOS]";
    @Serial
    private static final long serialVersionUID = -7580241079562697722L;
    private final Map<Integer, String> tokensByIndex = new HashMap<>();
    private final Map<String, Integer> indexByToken = new HashMap<>();

    private WordPieceVocab() {
    }

    @Override
    public boolean add(WordPiece wordPiece) {
        boolean wordAdded = super.add(wordPiece);
        wordPiece.wordIndex = this.size() - 1;
        tokensByIndex.put(wordPiece.wordIndex, wordPiece.token);
        indexByToken.put(wordPiece.token, wordPiece.wordIndex);
        return wordAdded;
    }

    public Optional<String> getTokenByIndex(int index) {
        return ofNullable(tokensByIndex.get(index));
    }

    public Optional<Integer> getTokenIndex(String token) {
        return ofNullable(indexByToken.get(token));
    }

    public InputStream getTokenReader() {
        return new CharSequenceInputStream(join("\n", indexByToken.keySet()), DEFAULT_CHAR_SET);
    }

    public static WordPieceVocab loadVocab(InputStream inputStream) {
        try {
            WordPieceVocab vocab = new WordPieceVocab();
            readLines(inputStream, DEFAULT_CHAR_SET).stream()
                    .map(token -> new WordPiece(token, token.startsWith(WORD_CONJUNCTION_SYMBOL)))
                    .forEach(vocab::add);
            return vocab;
        } catch (Exception e) {
            LOG.error("Got exception while loading a WordPieceVocab", e);
            throw e;
        }
    }

    /**
     * Representation of a token used in <a href="https://arxiv.org/pdf/1609.08144v2.pdf">Wordpiece Model</a>
     */
    public static class WordPiece implements Serializable {
        @Serial
        private static final long serialVersionUID = 5827241291787556930L;
        private final String token;
        private final boolean standaloneWord;
        private int wordIndex;

        public WordPiece(@Nonnull String token, boolean standaloneWord) {
            this.token = token;
            this.standaloneWord = standaloneWord;
        }

        public String getToken() {
            return token;
        }

        public boolean isStandaloneWord() {
            return standaloneWord;
        }
    }
}