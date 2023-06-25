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
import static org.tarik.utils.CommonUtils.readLines;

/**
 * Represents a set as a vocabulary of Part-Of-Speech (POS) Tags in {@link String} format.
 * Each tag has a look-up based on itself and its index. The latter is used by the model during training.
 */
public class PosTagsVocab extends HashSet<String> implements Serializable {
    private static final Logger LOG = LoggerFactory.getLogger(PosTagsVocab.class);
    public static final Charset DEFAULT_CHAR_SET = UTF_8;
    public static final String PADDING_TAG = "PAD";
    public static final String UNKNOWN_TAG = "UNKNOWN";
    @Serial
    private static final long serialVersionUID = -3785883839632301458L;
    private final Map<Integer, String> tagsByIndex = new HashMap<>();
    private final Map<String, Integer> indexByTag = new HashMap<>();

    private PosTagsVocab() {
    }

    @Override
    public boolean add(String tag) {
        boolean wordAdded = super.add(tag);
        int newIndex = this.size()-1;
        tagsByIndex.put(newIndex, tag);
        indexByTag.put(tag, newIndex);
        return wordAdded;
    }

    public Optional<String> getTagByIndex(int index){
        return ofNullable(tagsByIndex.get(index));
    }

    public boolean hasTag(String tag){
        return indexByTag.containsKey(tag);
    }

    public Optional<Integer> getTagIndex(String token){
        return ofNullable(indexByTag.get(token));
    }

    public InputStream getTokenReader(){
        return new CharSequenceInputStream(join("\n", indexByTag.keySet()), DEFAULT_CHAR_SET);
    }

    public static PosTagsVocab loadVocab(InputStream inputStream)  {
        try{
            PosTagsVocab vocab = new PosTagsVocab();
            vocab.addAll(readLines(inputStream, DEFAULT_CHAR_SET));
            return vocab;
        } catch (Exception e){
            LOG.error("Got exception while loading a POS Tag Vocab", e);
            throw e;
        }
    }
}