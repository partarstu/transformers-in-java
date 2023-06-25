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

import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import org.apache.commons.lang3.StringUtils;

import java.util.Collection;
import java.util.List;
import java.util.Properties;
import java.util.stream.Stream;

import static com.google.common.collect.ImmutableList.toImmutableList;

/**
 * A simple utility class which splits the specified text corpus into sentences. Uses {@link StanfordCoreNLP}
 * behind the scenes.
 */
public class SentenceSplitter {
    private static final Properties PROPERTIES = initializeProps();
    private static final StanfordCoreNLP pipeline = new StanfordCoreNLP(PROPERTIES);

    /**
     * Splits the specified text corpus into sentences.
     *
     * @param corpus the original text corpus which needs to be split
     * @return list containing the resulting sentences
     */

    public static List<String> breakCorpusIntoSentences(String corpus) {
        return getSentencesStream(corpus)
                .collect(toImmutableList());
    }

    /**
     * Splits the specified text corpus into sentences.
     *
     * @param corpus the original text corpus which needs to be split
     * @return {@link Stream} containing the resulting sentences
     */
    public static synchronized Stream<String> getSentencesStream(String corpus) {
        return Stream.of(corpus)
                .map(pipeline::processToCoreDocument)
                .map(CoreDocument::sentences)
                .flatMap(Collection::stream)
                .map(CoreSentence::text)
                .filter(StringUtils::isNotBlank)
                .map(String::toLowerCase);
    }

    private static Properties initializeProps() {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit");
        return props;
    }
}