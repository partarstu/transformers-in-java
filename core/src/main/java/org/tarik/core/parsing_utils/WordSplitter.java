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

package org.tarik.core.parsing_utils;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import org.apache.commons.lang3.StringUtils;

import java.util.Collection;
import java.util.List;
import java.util.Properties;
import java.util.stream.Stream;

import static com.google.common.collect.ImmutableList.toImmutableList;

/**
 * A simple utility class which splits the specified text corpus into words. Uses {@link StanfordCoreNLP}
 * behind the scenes.
 */
public class WordSplitter {
    private static final Properties PROPERTIES = initializeProps();
    private static final StanfordCoreNLP pipeline = new StanfordCoreNLP(PROPERTIES);

    /**
     * Splits the specified text corpus into tokens (words). Each token is lower-cased by default.
     *
     * @param corpus the original text corpus which needs to be split
     * @return list containing the resulting tokens (words)
     */
    public static List<String> breakCorpusIntoWords(String corpus) {
        return getWordsStream(corpus, true)
                .collect(toImmutableList());
    }

    /**
     * Splits the specified text corpus into tokens (words). Casing is done based on the specified flag.
     *
     * @param corpus       the original text corpus which needs to be split
     * @param useLowerCase if each token should be lower-cased
     * @return list containing the resulting tokens (words)
     */
    public static List<String> breakCorpusIntoWords(String corpus, boolean useLowerCase) {
        return getWordsStream(corpus, useLowerCase)
                .collect(toImmutableList());
    }

    /**
     * Splits the specified text corpus into tokens (words). Casing is done based on the specified flag.
     *
     * @param corpus       the original text corpus which needs to be split
     * @param useLowerCase if each token should be lower-cased
     * @return {@link Stream} containing the resulting tokens (words)
     */
    public static synchronized Stream<String> getWordsStream(String corpus, boolean useLowerCase) {
        return Stream.of(corpus)
                .map(pipeline::processToCoreDocument)
                .map(CoreDocument::tokens)
                .flatMap(Collection::stream)
                .map(CoreLabel::originalText)
                .filter(StringUtils::isNotBlank)
                .map(word -> useLowerCase ? word.toLowerCase() : word);
    }

    private static Properties initializeProps() {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit");
        return props;
    }
}