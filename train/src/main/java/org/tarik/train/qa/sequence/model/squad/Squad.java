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

package org.tarik.train.qa.sequence.model.squad;

import java.util.ArrayList;
import java.util.List;

public class Squad implements Cloneable{
    private String version;
    private List<QaData> data;

    public String getVersion() {
        return version;
    }

    public List<QaData> getData() {
        return data;
    }

    public void truncateQaData(int fromIndex){
        this.data =  this.data.subList(fromIndex, this.data.size());
    }

    public Squad clone() throws CloneNotSupportedException {
        Squad clone = (Squad)super.clone();
        clone.data = new ArrayList<>(this.data);
        return clone;
    }
}