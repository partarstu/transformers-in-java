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

import com.mongodb.MongoClientSettings;
import com.mongodb.client.MongoClients;
import dev.morphia.Datastore;
import dev.morphia.query.FindOptions;
import dev.morphia.query.experimental.filters.Filter;

import java.util.List;

import static dev.morphia.Morphia.createDatastore;
import static java.util.concurrent.TimeUnit.MINUTES;
import static java.util.concurrent.TimeUnit.SECONDS;

public class WikiDatastore {
    private static final Datastore dataStore = initializeDatastore();

    private static Datastore initializeDatastore() {
        MongoClientSettings mongoClientSettings = MongoClientSettings.builder()
                .applyToClusterSettings(settings ->
                        settings.serverSelectionTimeout(60, SECONDS))
                .applyToConnectionPoolSettings(settings ->
                        settings.maintenanceFrequency(10, MINUTES))
                .applyToServerSettings(settings ->
                        settings.heartbeatFrequency(30, SECONDS)
                                .minHeartbeatFrequency(15, SECONDS))
                .applyToSocketSettings(settings -> settings.connectTimeout(20, SECONDS))
                .build();

        final Datastore datastore = createDatastore(MongoClients.create(mongoClientSettings), "wiki_en");
        datastore.getMapper().mapPackage("org.tarik.core.db.model");
        //datastore.ensureIndexes();
        return datastore;
    }

    public static <T extends WikiArticle> List<T> fetchArticlesFromDb(Class<T> clazz, Filter... filters) {
        return dataStore.find(clazz).filter(filters).iterator().toList();
    }

    public static <T extends WikiArticle> List<T> fetchArticlesFromDb(Class<T> clazz, int limit, Filter... filters) {
        return dataStore.find(clazz).filter(filters)
                .iterator(new FindOptions().limit(limit))
                .toList();
    }
}