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

package org.tarik.train;

import com.google.common.collect.ImmutableSet;
import org.tarik.core.data.IDataProvider;
import org.tarik.train.db.model.wiki.WikiTextArticle;

import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static com.google.common.base.Preconditions.checkArgument;
import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;
import static org.tarik.core.parsing_utils.SentenceSplitter.breakCorpusIntoSentences;
import static org.tarik.utils.CommonUtils.normalizeWikiEntityName;

/**
 * A sample implementation of {@link IDataProvider} which fetches the passages using the {@link WikiTextArticle} as the data adaptor.
 * Behind the scenes there's a MongoDB instance running which stores the actual entities.
 */
public class WikiArticlesContentProvider implements IDataProvider {
    private static final Set<String> CATEGORIES_TO_EXCLUDE = ImmutableSet.of("see also", "notes", "references",
            "citations", "bibliography", "further reading", "external links");
    private final Supplier<Collection<WikiTextArticle>> articlesFetcher;
    private final boolean fetchArticleSummaryOnly;

    public WikiArticlesContentProvider(Supplier<Collection<WikiTextArticle>> articlesFetcher,
                                       boolean fetchArticleSummaryOnly) {
        super();
        this.articlesFetcher = articlesFetcher;
        this.fetchArticleSummaryOnly = fetchArticleSummaryOnly;
    }

    @Override
    public List<String> getPassages(Function<List<String>, Boolean> isLimitReachedFunction) {
        List<String> fetchedSentences = new LinkedList<>();
        while (!isLimitReachedFunction.apply(fetchedSentences)) {
            extractWikiPageParagraphsByCategory(articlesFetcher.get(), fetchArticleSummaryOnly)
                    .forEach(passageByTitle -> fetchedSentences.addAll(passageByTitle.values()
                            .stream().flatMap(Collection::stream).toList()));
        }
        return fetchedSentences;
    }

    private static Stream<Map<String, List<String>>> extractWikiPageParagraphsByCategory(Collection<WikiTextArticle> wikiTextArticles,
                                                                                         boolean onlyArticleSummary) {
        checkArgument(wikiTextArticles != null,
                "Collection of wiki articles which need to be learned by the agent can't be NULL");

        return wikiTextArticles.stream()
                .map(article -> getArticleParagraphsByCategory(article, onlyArticleSummary))
                .filter(paragraphsByCategory -> !paragraphsByCategory.isEmpty());
    }

    /**
     * Takes the wiki article and converts it into paragraphs based on categories. Each paragraph will be split into sentences.
     *
     * @param wikiTextArticle    wiki text article which needs to be split
     * @param onlyArticleSummary if only the contents of article's summary should be retrieved, otherwise all contents
     * @return - a map of paragraph sentences split by the category
     */
    private static Map<String, List<String>> getArticleParagraphsByCategory(WikiTextArticle wikiTextArticle,
                                                                            boolean onlyArticleSummary) {
        String articleName = normalizeWikiEntityName(wikiTextArticle.getName()).toLowerCase();
        HashMap<String, List<String>> paragraphContentByCategory = new HashMap<>();
        addCategoryWithParagraphSentences(paragraphContentByCategory, articleName, wikiTextArticle.getTextSummary());
        if (!onlyArticleSummary) {
            wikiTextArticle.getParagraphs().forEach(paragraph ->
                    addCategoryWithParagraphSentences(paragraphContentByCategory, paragraph.getParagraphName(),
                            paragraph.getContent()));
        }
        return paragraphContentByCategory;
    }

    private static void addCategoryWithParagraphSentences(Map<String, List<String>> collector, String category,
                                                          String paragraph) {
        String categoryName = normalizeWikiEntityName(category);
        if (!paragraph.isBlank() && !categoryName.isBlank() && paragraph.trim().length() > categoryName.length() &&
                !CATEGORIES_TO_EXCLUDE.contains(categoryName)) {
            paragraph = paragraph.replace("(; )", "")
                    .replace("()", "");
            List<String> sentences = stream(paragraph.split("\n\n"))
                    .map(WikiArticlesContentProvider::getNormalizedParagraphSentences)
                    .flatMap(Collection::stream)
                    .filter(sentence -> sentence.length() > categoryName.length())
                    .collect(toList());
            if (!sentences.isEmpty()) {
                collector.putIfAbsent(category, sentences);
            }
        }
    }

    private static List<String> getNormalizedParagraphSentences(String originalText) {
        return breakCorpusIntoSentences(originalText.replace("\n", " "))
                .stream()
                .filter(sentence -> !sentence.contains("also refer to"))
                .filter(sentence -> !sentence.contains("may refer to"))
                .collect(toList());
    }
}