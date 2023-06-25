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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.google.common.collect.ImmutableSet;
import com.google.common.reflect.TypeToken;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.lang.reflect.Type;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileTime;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.primitives.Bytes.toArray;
import static java.lang.Runtime.getRuntime;
import static java.lang.String.format;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.Files.*;
import static java.time.Duration.between;
import static java.time.Instant.now;
import static java.util.Arrays.copyOf;
import static java.util.Arrays.fill;
import static java.util.Objects.requireNonNull;
import static java.util.Optional.*;
import static java.util.concurrent.TimeUnit.SECONDS;
import static java.util.regex.Pattern.compile;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;
import static org.jsoup.Jsoup.parse;

/**
 * Utility class for facilitating the development.
 */
public class CommonUtils {
    public static final Charset DEFAULT_CHAR_SET = UTF_8;
    public static final String SEPARABLE_PUNCTUATIONS_PATTERN = ("[\\[\\]\\\\]+");
    public static final String REPLACEABLE_PUNCTUATIONS_PATTERN = ("[()|;{}><]+");

    private static final Logger LOG = LoggerFactory.getLogger(CommonUtils.class);

    public static Optional<String> decodeAsHtmlText(String value) {
        requireNonNull(value);
        try {
            return of(parse(value).text());
        } catch (Exception e) {
            return empty();
        }
    }

    public static String normalizeWikiEntityName(String originalName) {
        return originalName.replaceAll("_", " ")
                .replaceAll("disambiguation.*", "")
                .replaceAll("\\(.*?\\)", "")
                .replaceAll(SEPARABLE_PUNCTUATIONS_PATTERN, " ")
                .replaceAll(REPLACEABLE_PUNCTUATIONS_PATTERN, "")
                .trim();
    }

    public static Optional<String> getMatchingGroup(String value, String regex, int groupIndex) {
        requireNonNull(value);
        requireNonNull(regex);
        var matcher = compile(regex).matcher(value);
        if (matcher.find() && groupIndex <= matcher.groupCount()) {
            return of(matcher.group(groupIndex));
        } else {
            return empty();
        }
    }

    public static Optional<FileTime> getFileLastModifiedDateTime(Path path) {
        try {
            return of(getLastModifiedTime(path));
        } catch (Exception e) {
            LOG.error("Couldn't retrieve the last modified date of the file", e);
            return empty();
        }
    }

    public static String getCommandOutput(Process p) {
        return new BufferedReader(new InputStreamReader(p.getInputStream()))
                .lines()
                .collect(joining("\n"));
    }

    public static void sleepMillis(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            LOG.error("Sleep has been interrupted", e);
            throw new IllegalStateException(e);
        }
    }

    public static void deleteFiles(Collection<Path> files) {
        try {
            for (Path file : files) {
                deleteIfExists(file);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static String readResourceFile(Class<?> clazz, String fileName) {
        return readResourceFile(clazz, fileName, DEFAULT_CHAR_SET);
    }

    public static List<String> readLines(InputStream inputStream) {
        return readLines(inputStream, DEFAULT_CHAR_SET);
    }

    public static List<String> readLines(InputStream inputStream, Charset charset) {
        return new BufferedReader(new InputStreamReader(inputStream, charset)).lines().collect(toList());
    }

    public static String readResourceFile(Class<?> clazz, String fileName, Charset charset) {
        try {
            var filePath = ofNullable(clazz.getResource(fileName)).map(CommonUtils::toUri).map(Paths::get)
                    .orElseThrow(() -> new IllegalStateException(format("Couldn't load %s file as a resource of %s", fileName, clazz)));
            return readString(filePath);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static URI toUri(URL url) {
        try {
            return url.toURI();
        } catch (URISyntaxException e) {
            throw new IllegalArgumentException(e);
        }
    }

    public static long getDurationInSeconds(Instant start) {
        return between(start, now()).toSeconds();
    }

    public static long getDurationInMinutes(Instant start) {
        return between(start, now()).toMinutes();
    }

    public static long getDurationInMillis(Instant start) {
        return between(start, now()).toMillis();
    }

    public static long getApproximateFreeMemoryMb() {
        return (getRuntime().maxMemory() - getRuntime().totalMemory() + getRuntime().freeMemory()) / 1024 / 1024;
    }

    public static void sleepSeconds(int seconds) {
        sleepMillis(SECONDS.toMillis(seconds));
    }

    public static ImmutableSet<String> getWordBoundRegexes(Collection<String> originalRegexes) {
        return originalRegexes.stream()
                .map(originalRegex -> "\\b" + originalRegex + "\\b")
                .collect(toImmutableSet());
    }

    public static <T> ImmutableSet<T> getMergedSet(Collection<T> collection1, Collection<T> collection2) {
        return Stream.concat(collection1.stream(), collection2.stream())
                .collect(toImmutableSet());
    }

    public static <T> T instantiateClass(Class<T> clazz) {
        try {
            return clazz.getDeclaredConstructor().newInstance();
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    public static String prettyPrintExceptionTrace(Throwable t) {
        return format("%s\n%s", t, Arrays.toString(t.getStackTrace()));
    }

    public static String serializeIntoJsonWithJackson(Object object) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            return objectMapper.writeValueAsString(object);
        } catch (JsonProcessingException e) {
            LOG.error("Couldn't serialize an object into JSON", e);
            return "";
        }
    }

    public static String serializeIntoJsonWithGson(Object object) {
        try {
            Gson gson = new Gson();
            return gson.toJson(object);
        } catch (Exception e) {
            LOG.error("Couldn't serialize an object into JSON", e);
            return "";
        }
    }

    public static <T> Optional<T> deserializeObjectFromJsonWithJackson(String jsonString, Class<T> clazz) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            return ofNullable(objectMapper.readValue(jsonString, clazz));
        } catch (JsonProcessingException e) {
            LOG.error(format("Couldn't deserialize an object of type %s from JSON", clazz), e);
            return empty();
        }
    }

    public static <T, U> Optional<Map<T, U>> deserializeMapFromJsonWithJackson(String jsonString, Class<T> keyClass,
                                                                               Class<U> valueClass) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            return ofNullable(objectMapper.readValue(jsonString, new TypeReference<>() {
            }));
        } catch (JsonProcessingException e) {
            LOG.error(format("Couldn't deserialize an object as a Map<%s, %s> from JSON",
                    keyClass.getTypeName(), valueClass.getTypeName()), e);
            return empty();
        }
    }

    public static <T, U> Optional<Map<T, U>> deserializeMapFromJsonWithJackson(String jsonString) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            return ofNullable(objectMapper.readValue(jsonString, new TypeReference<>() {
            }));
        } catch (JsonProcessingException e) {
            return empty();
        }
    }

    public static <T> Optional<List<T>> deserializeListFromJsonWithJackson(String jsonString, Class<T> clazz) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            CollectionType collectionType = objectMapper.getTypeFactory().constructCollectionType(List.class, clazz);
            return ofNullable(objectMapper.readValue(jsonString, collectionType));
        } catch (JsonProcessingException e) {
            LOG.error(format("Couldn't deserialize the list of objects of type %s from JSON", clazz), e);
            return empty();
        }
    }

    public static <T> Optional<List<T>> deserializeCollectionFromJsonWithGson(String jsonString, Class<T> clazz) {
        try {
            GsonBuilder gsonBuilder = new GsonBuilder();
            Type collectionType = new TypeToken<List<T>>() {
            }.getType();
            Gson gson = gsonBuilder.create();
            return ofNullable(gson.fromJson(jsonString, collectionType));
        } catch (JsonSyntaxException e) {
            LOG.error(format("Couldn't deserialize the list of objects of type %s from JSON", clazz), e);
            return empty();
        }
    }

    public static void executeWithReadLock(ReadWriteLock readWriteLock, Runnable runnable) {
        readWriteLock.readLock().lock();
        try {
            runnable.run();
        } finally {
            readWriteLock.readLock().unlock();
        }
    }

    public static <T> T executeWithReadLock(ReadWriteLock readWriteLock, Supplier<T> supplier) {
        readWriteLock.readLock().lock();
        try {
            return supplier.get();
        } finally {
            readWriteLock.readLock().unlock();
        }
    }

    public static void logException(Logger logger, Throwable throwable, String operationDescription) {
        logger.error("Caught exception while {}. Original Exception: {}",
                operationDescription, prettyPrintExceptionTrace(throwable));
    }

    public static byte[] copyByteListExtendingWithPadding(List<Byte> originalByteList, int newSize, byte paddingValue) {
        byte[] extendedArray = copyOf(toArray(originalByteList), newSize);
        if (extendedArray.length > originalByteList.size()) {
            fill(extendedArray, originalByteList.size() - 1, newSize, paddingValue);
        }
        return extendedArray;
    }

    public static int[] copyIntegerListExtendingWithPadding(List<Integer> originalIntegerList, int newSize,
                                                            int paddingValue) {
        int[] extendedArray = copyOf(originalIntegerList.stream()
                        .mapToInt(Integer::intValue)
                        .toArray(),
                newSize);
        if (extendedArray.length > originalIntegerList.size()) {
            fill(extendedArray, originalIntegerList.size() - 1, newSize, paddingValue);
        }
        return extendedArray;
    }

    public static String getCounterEnding(int counter) {
        int base = counter > 20 ? counter % 10 : counter;
        return counter + switch (base) {
            case 1 -> "st";
            case 2 -> "nd";
            case 3 -> "rd";
            default -> "th";
        };
    }

}