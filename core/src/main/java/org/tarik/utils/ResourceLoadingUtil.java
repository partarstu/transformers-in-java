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

package org.tarik.utils;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;
import static java.lang.String.format;
import static java.util.Objects.requireNonNull;
import static org.apache.commons.lang3.StringUtils.isNotEmpty;
import static org.tarik.utils.CommonUtils.readLines;

/**
 * Utility class for facilitating the loading of resource files.
 */
public class ResourceLoadingUtil {
    public static InputStream getResourceFileStream(String fileName, Class<?> clazz) {
        checkArgument(isNotEmpty(fileName), "Resource file name must be specified");
        return requireNonNull(clazz.getClassLoader().getResourceAsStream(fileName), format("Resource file %s doesn't exist", fileName));
    }

    public static List<String> getResourceFileLines(String fileName, Class<?> clazz) {
        checkArgument(isNotEmpty(fileName), "Resource file name must be specified");
        try (InputStream is = requireNonNull(clazz.getClassLoader().getResourceAsStream(fileName))) {
            return readLines(is);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

}