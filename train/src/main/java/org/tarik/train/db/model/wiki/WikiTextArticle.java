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

package org.tarik.train.db.model.wiki;

import com.google.common.collect.ImmutableList;
import dev.morphia.annotations.Entity;

import java.util.LinkedList;
import java.util.List;

@Entity
public class WikiTextArticle extends WikiArticle {
    private String textSummary;
    private final List<WikiTextParagraph> paragraphs = new LinkedList<>();

    // For reflection
    private WikiTextArticle() {
    }

    public WikiTextArticle(String name) {
        super(name);
    }

    public void setTextSummary(String textSummary) {
        this.textSummary = textSummary;
    }

    public void addParagraph(WikiTextParagraph wikiTextParagraph) {
        paragraphs.add(wikiTextParagraph);
    }

    public String getName() {
        return name;
    }

    public String getTextSummary() {
        return textSummary;
    }

    public List<WikiTextParagraph> getParagraphs() {
        return ImmutableList.copyOf(paragraphs);
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("WikiTextArticle{");
        sb.append("name='").append(name).append('\'');
        sb.append(", textSummary='").append(textSummary).append('\'');
        sb.append(", paragraphs=").append(paragraphs);
        sb.append('}');
        return sb.toString();
    }
}