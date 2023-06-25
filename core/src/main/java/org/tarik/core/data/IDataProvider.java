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

package org.tarik.core.data;

import java.util.List;
import java.util.function.Function;

/**
 * Feeds the data into the model during training.
 */
public interface IDataProvider {
    /**
     * Provides a new batch of text passages. Those could be sentences or paragraphs, depends on the model. The amount of the fetched
     * passages should be limited by the function, otherwise memory-related issues could arise.
     *
     * @param isLimitReachedFunction function which decides if fetching should be stopped based on the list of already fetched passages.
     * @return list of the fetched text passages
     */
    List<String> getPassages(Function<List<String>, Boolean> isLimitReachedFunction);
}