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

package org.tarik.utils.visualisation.charts;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberTickUnitSource;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ui.RectangleInsets;
import org.jfree.chart.ui.UIUtils;
import org.jfree.data.xy.DefaultTableXYDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;

import javax.swing.*;
import java.awt.*;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.jfree.chart.ChartFactory.createScatterPlot;

/**
 * A simple plotter which aims to visualize the convergence process for the model. It collects and plots the XY-series data for each
 * accuracy type it has been initialized with.
 */
public class ConvergenceChartPlotter extends JFrame {
    private final DefaultTableXYDataset dataset = new DefaultTableXYDataset();
    private final Map<String, XYSeries> dataByAccuracyType = new LinkedHashMap<>();

    /**
     * Creates a new plotter instance.
     *
     * @param title         the frame title
     * @param accuracyTypes accuracy types
     */
    private ConvergenceChartPlotter(String title, Collection<String> accuracyTypes) {
        super(title);
        accuracyTypes
                .forEach(dataType -> dataByAccuracyType.putIfAbsent(dataType, new XYSeries(dataType, true, false)));
        dataByAccuracyType.values().forEach(dataset::addSeries);
        ChartPanel chartPanel = (ChartPanel) createPanel();
        chartPanel.setPreferredSize(new Dimension(1000, 800));
        setContentPane(chartPanel);
    }

    /**
     * Returns a panel containing the content of the plotter.
     *
     * @return A panel containing the content
     */
    private JPanel createPanel() {
        JFreeChart chart = createChart(dataset);
        ChartPanel panel = new ChartPanel(chart, false);
        panel.setFillZoomRectangle(true);
        panel.setMouseWheelEnabled(true);
        return panel;
    }

    /**
     * Creates a line chart with the supplied dataset.
     *
     * @param dataset the dataset.
     * @return A line chart.
     */
    private static JFreeChart createChart(XYDataset dataset) {
        JFreeChart chart = createScatterPlot("Convergence", "Iterations", "Accuracy", dataset);
        chart.setBackgroundPaint(Color.WHITE);

        XYPlot plot = (XYPlot) chart.getPlot();
        plot.setBackgroundPaint(Color.LIGHT_GRAY);
        plot.setDomainGridlinePaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.WHITE);
        plot.setAxisOffset(new RectangleInsets(5.0, 5.0, 5.0, 5.0));
        plot.setDomainCrosshairVisible(true);
        plot.setRangeCrosshairVisible(true);

        XYItemRenderer r = plot.getRenderer();
        if (r instanceof XYLineAndShapeRenderer xyLineAndShapeRenderer) {
            xyLineAndShapeRenderer.setDefaultShapesVisible(true);
            xyLineAndShapeRenderer.setDefaultShapesFilled(true);
            xyLineAndShapeRenderer.setDrawSeriesLineAsPath(true);
            xyLineAndShapeRenderer.setDefaultLinesVisible(true);
            xyLineAndShapeRenderer.setDrawOutlines(true);
        }

        ValueAxis domainAxis = plot.getDomainAxis();
        plot.getRangeAxis().setUpperBound(100);
        domainAxis.setStandardTickUnits(new NumberTickUnitSource(true));

        return chart;
    }


    /**
     * Adds the data to the series and plots it onto the chart.
     *
     * @param accuracyType  type of the accuracy for which the data should be added
     * @param iteration     training iteration for which the data should be added
     * @param modelAccuracy value of the accuracy
     */
    public void addData(String accuracyType, int iteration, double modelAccuracy) {
        dataByAccuracyType.computeIfPresent(accuracyType, (type, data) -> {
            data.addOrUpdate(iteration, modelAccuracy);
            return data;
        });
    }

    /**
     * Static factory method for creating a new plotter for the specified accuracy types with the corresponding title.
     *
     * @param title         title of new plotter
     * @param accuracyTypes types of the accuracy for which the data will be plotted
     * @return new instance of the plotter
     */
    public static ConvergenceChartPlotter newChartPlotter(String title, Collection<String> accuracyTypes) {
        ConvergenceChartPlotter plotter = new ConvergenceChartPlotter(title, accuracyTypes);
        plotter.pack();
        UIUtils.centerFrameOnScreen(plotter);
        plotter.setVisible(true);
        return plotter;
    }
}