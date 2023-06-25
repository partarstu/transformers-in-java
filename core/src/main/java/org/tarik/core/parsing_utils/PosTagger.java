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
import java.util.Properties;
import java.util.stream.Stream;

/**
 * A simple utility class which provides Part-Of-Speech (POS) Tags for the specified text corpus.
 */
public class PosTagger {
    private static final Properties PROPERTIES = initializeProps();
    private static final StanfordCoreNLP pipeline = new StanfordCoreNLP(PROPERTIES);

    /**
     * Tokenizes the specified text corpus and converts each token into the {@link Stream} of POS-Tags. Uses {@link StanfordCoreNLP}
     * behind the scenes.
     *
     * @param corpus the original text corpus which needs to be tagged
     * @return {@link Stream} containing the corresponding POS-Tags
     */
    public static synchronized Stream<String> getPosTagsStream(String corpus) {
        return Stream.of(corpus)
                .map(pipeline::processToCoreDocument)
                .map(CoreDocument::tokens)
                .flatMap(Collection::stream)
                .map(CoreLabel::tag)
                .filter(StringUtils::isNotBlank);
    }

    private static Properties initializeProps() {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos");
        return props;
    }
}