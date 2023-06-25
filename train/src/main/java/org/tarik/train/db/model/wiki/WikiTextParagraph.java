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

import dev.morphia.annotations.Embedded;

import java.util.ArrayList;
import java.util.List;

@Embedded
public class WikiTextParagraph {
    private String paragraphName;
    private final List<String> content = new ArrayList<>();

    // For reflection
    private WikiTextParagraph() {
    }

    public WikiTextParagraph(String paragraphName) {
        this.paragraphName = paragraphName;
    }

    public void addNewContent(StringBuilder newContent){
        content.add(newContent.toString().trim());
    }

    public String getParagraphName() {
        return paragraphName;
    }

    public String getContent() {
        return content.toString();
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("WikiTextParagraph{");
        sb.append("paragraphName='").append(paragraphName).append('\'');
        sb.append(", content=").append(content);
        sb.append('}');
        return sb.toString();
    }
}