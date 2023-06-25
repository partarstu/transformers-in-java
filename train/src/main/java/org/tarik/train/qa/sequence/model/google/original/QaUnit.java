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

package org.tarik.train.qa.sequence.model.google.original;

import com.google.gson.annotations.SerializedName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static java.lang.String.join;
import static java.util.Arrays.copyOfRange;
import static java.util.Arrays.stream;
import static org.jsoup.Jsoup.parse;

public class QaUnit {
    private static final Logger LOG = LoggerFactory.getLogger(QaUnit.class);

    @SerializedName("document_text")
    private String documentText;

    @SerializedName("question_text")
    private String question;

    @SerializedName("annotations")
    private List<Annotation> annotations;

    @SerializedName("long_answer_candidates")
    private List<AnswerRange> longAnswerCandidates;

    public boolean isYesNoAnswer() {
        return !annotations.get(0).getYesNo().equals("NONE");
    }

    public String getDocumentText() {
        return documentText;
    }

    public String getQuestion() {
        return question;
    }

    public List<Annotation> getAnnotations() {
        return annotations;
    }

    public List<AnswerRange> getLongAnswerCandidates() {
        return longAnswerCandidates;
    }

    public void clear(){
        this.documentText=null;
        annotations.clear();
        longAnswerCandidates.clear();
    }
}