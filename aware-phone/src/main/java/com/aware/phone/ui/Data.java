package com.aware.phone.ui;


import android.app.ProgressDialog;
import android.content.ContentValues;
import android.content.Context;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.database.sqlite.SQLiteException;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.phone.R;
import com.aware.providers.Aware_Provider;
import com.aware.providers.ESM_Provider;
import com.aware.ui.esms.ESM_Question;
import com.aware.utils.Jdbc;
import com.jjoe64.graphview.DefaultLabelFormatter;
import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.LegendRenderer;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.LineGraphSeries;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.sql.Timestamp;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;

public class Data extends Aware_Activity{

    private Calendar dayBefore;
    private LinearLayout container;
    private LayoutInflater inflater;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.live_data);
//        setContentView(R.layout.data);
        dayBefore = Calendar.getInstance();

        container = findViewById(R.id.graph_layout_container);
        inflater = LayoutInflater.from(this);
        visualizeWeekESM();
    }

    /**
     * Retrieves and visualizes EMS responses for the week.
     * Presents scale rating in line graphs.
     * References: https://github.com/jjoe64/GraphView
     */
    private void visualizeWeekESM() {
        JSONArray weekData = getDummyESMData();
        // retrieveWeekESMDataFromLocalDb();
        // new queryWeekDataFromCloudDB().execute("");

        // Get all ESM questions that have been configured
        JSONArray questions = new JSONArray();
        Cursor study = Aware.getStudy(getApplicationContext(),
                Aware.getSetting(getApplicationContext(), Aware_Preferences.WEBSERVICE_SERVER));
        if (study != null && study.moveToFirst()) {
            try {
                JSONObject localConfig = new JSONObject(study.getString(
                        study.getColumnIndex(Aware_Provider.Aware_Studies.STUDY_CONFIG)));
                questions = new JSONArray(localConfig.getString("questions"));
            } catch (JSONException e) {
                e.printStackTrace();
            }
            study.close();
        }

        ArrayList<ArrayList<DataPoint>> questionDataPoints = new ArrayList<>();
        ArrayList<String> questionTitles = new ArrayList<>();
        try {
            // Pre-saves the titles for all ESM questions for display
            for (int i=0; i<questions.length(); i++) {
                questionDataPoints.add(new ArrayList<DataPoint>());
                JSONObject questionDesc = questions.getJSONObject(i);
                questionTitles.add(questionDesc.getString(ESM_Question.esm_title));
            }
            // Segregates retrieved ESM responses based on corresponding questions
            ArrayList<String> onsetTimes = new ArrayList<>();
            ArrayList<Timestamp> onsetTimestamps = new ArrayList<>();
            ArrayList<String> wakeTimes = new ArrayList<>();
            ArrayList<Timestamp> wakeTimestamps = new ArrayList<>();

            for (int i=0; i < weekData.length(); i++) {
                JSONObject data = weekData.getJSONObject(i);
                JSONObject dataEsm = new JSONObject(data.getString(ESM_Provider.ESM_Data.JSON));
                if (dataEsm.getInt(ESM_Question.esm_type) == 6) {
                    ArrayList questionArray = questionDataPoints.get(dataEsm.getInt("id")-1);
                    Timestamp timestamp = new Timestamp(data.getLong((ESM_Provider.ESM_Data.TIMESTAMP)));
                    Date curDate = new Date(timestamp.getTime());
                    int response = data.getInt(ESM_Provider.ESM_Data.ANSWER);
                    questionArray.add(new DataPoint(curDate, response));
                } else if (dataEsm.getInt(ESM_Question.esm_type) == 12 && dataEsm.getInt("id") == 3) {
                    onsetTimes.add(data.getString(ESM_Provider.ESM_Data.ANSWER));
                    onsetTimestamps.add(new Timestamp(data.getLong((ESM_Provider.ESM_Data.ANSWER_TIMESTAMP))));
                } else if (dataEsm.getInt(ESM_Question.esm_type) == 12 && dataEsm.getInt("id") == 4) {
                    wakeTimes.add(data.getString(ESM_Provider.ESM_Data.ANSWER));
                    wakeTimestamps.add(new Timestamp(data.getLong((ESM_Provider.ESM_Data.ANSWER_TIMESTAMP))));
                }
            }
            // Computes sleep duration based on self-reported sleep onset and wake times
            ArrayList<Date> dates = new ArrayList<>();
            ArrayList<Float> sleepDurations = new ArrayList<>();
            for (int i=0; i<onsetTimes.size(); i++) {
                Date startDate = new Date(onsetTimestamps.get(i).getTime());
                for (int j=0; j<wakeTimes.size(); j++) {
                    Date endDate = new Date(wakeTimestamps.get(j).getTime());
                    int dateDiff = endDate.compareTo(startDate);
                    if (dateDiff == 0) {
                        String[] startHourMinute = onsetTimes.get(i).split(":");
                        int startHour = Integer.parseInt(startHourMinute[0]);
                        float startMinute = Float.parseFloat(startHourMinute[1])/60;
                        float floatStartHourMinute = startHour + startMinute;
                        String[] endHourMinute = wakeTimes.get(j).split(":");
                        int endHour = Integer.parseInt(endHourMinute[0]);
                        float endMinute = Float.parseFloat(endHourMinute[1])/60;
                        float floatEndHourMinute = endHour + endMinute;
                        float sleepDuration;
                        // Falls asleep past midnight
                        if (floatEndHourMinute > floatStartHourMinute) {
                            sleepDuration = floatEndHourMinute - floatStartHourMinute;
                        }
                        // Falls asleep before midnight
                        else if (floatStartHourMinute > floatEndHourMinute) {
                            sleepDuration = (24 - floatStartHourMinute) + floatEndHourMinute;
                        } else {
                            // Assumes that a mistake is made: correct to 12 hrs by default.
                            sleepDuration = 12;
                        }
                        // Assumes that a mistake is made (e.g., 00:00 was input as 12:00)
                        if (sleepDuration >= 15) {
                            sleepDuration = sleepDuration - 12;
                        }
                        dates.add(startDate);
                        sleepDurations.add(sleepDuration);
                        wakeTimes.remove(j);
                        wakeTimestamps.remove(j);
                        break;
                    }
                }
            }

            // Prepare a series graph for sleep duration
            if (sleepDurations.size() > 0) {
                DataPoint[] sleepDataPoints = new DataPoint[sleepDurations.size()];
                for (int i=0; i<sleepDurations.size(); i++) {
                    sleepDataPoints[i] = new DataPoint(dates.get(i), sleepDurations.get(i));
                }
                View graphView = inflater.inflate(R.layout.graph_view, null);
                GraphView graph = (GraphView) graphView.findViewById(R.id.esm_graph);
                TextView graphTitle = graphView.findViewById(R.id.esm_title);
                LineGraphSeries<DataPoint> series = new LineGraphSeries<DataPoint>(sleepDataPoints);
                graph.addSeries(series);
                graphTitle.setText("Reported Sleep Duration");

                // Adds graph dynamically into the container view
                LinearLayout.LayoutParams graphViewParams = new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
                graphViewParams.setMargins(0, 0, 0, 50);
                graphView.setLayoutParams(graphViewParams);
                // Ensure to display all date labels
                graph.getGridLabelRenderer().setNumHorizontalLabels(sleepDataPoints.length);
                // Reference: https://github.com/jjoe64/GraphView-Demos/blob/master/app/src/main/java/com/jjoe64/graphview_demos/examples/CustomLabelsGraph.java
                graph.getGridLabelRenderer().setLabelFormatter(new DefaultLabelFormatter() {
                    @Override
                    public String formatLabel(double value, boolean isValueX) {
                        if (isValueX) {
                            Calendar c = Calendar.getInstance();
                            c.setTimeInMillis((long)value);
                            int mMonth = c.get(Calendar.MONTH);
                            int mDay = c.get(Calendar.DAY_OF_MONTH);
                            return mDay + "/" + (mMonth+1);
                        } else {
                            return super.formatLabel(value, isValueX);
                        }
                    }
                });
                container.addView(graphView);
            }

            // Prepares a series graph using responses for each scale rating ESM question
            for (int i=0; i<questionDataPoints.size(); i++) {
                ArrayList<DataPoint> curQuestionPoints = questionDataPoints.get(i);
                if (curQuestionPoints.size() > 0) {
                    DataPoint[] curQuestionDataPoints = new DataPoint[curQuestionPoints.size()];
                    for (int j=0; j<curQuestionPoints.size(); j++) {
                        curQuestionDataPoints[j] = curQuestionPoints.get(j);
                    }
                    View graphView = inflater.inflate(R.layout.graph_view, null);
                    GraphView graph = (GraphView) graphView.findViewById(R.id.esm_graph);
                    TextView graphTitle = graphView.findViewById(R.id.esm_title);
                    LineGraphSeries<DataPoint> series = new LineGraphSeries<DataPoint>(curQuestionDataPoints);
                    graph.addSeries(series);
                    graphTitle.setText(questionTitles.get(i));

                    // Adds graph dynamically into the container view
                    LinearLayout.LayoutParams graphViewParams = new LinearLayout.LayoutParams(
                            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
                    graphViewParams.setMargins(0, 0, 0, 50);
                    graphView.setLayoutParams(graphViewParams);
                    // Ensure to display all date labels
                    graph.getGridLabelRenderer().setNumHorizontalLabels(curQuestionDataPoints.length);
                    graph.getGridLabelRenderer().setLabelFormatter(new DefaultLabelFormatter() {
                        @Override
                        public String formatLabel(double value, boolean isValueX) {
                            if (isValueX) {
                                Calendar c = Calendar.getInstance();
                                c.setTimeInMillis((long)value);
                                int mMonth = c.get(Calendar.MONTH);
                                int mDay = c.get(Calendar.DAY_OF_MONTH);
                                return mDay + "/" + (mMonth+1);
                            } else {
                                return super.formatLabel(value, isValueX);
                            }
                        }
                    });
                    container.addView(graphView);
                }
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    private JSONArray retrieveWeekESMDataFromLocalDb() {
        long endTimestamp = dayBefore.getTime().getTime();
        Calendar weekStart = dayBefore;
        weekStart.add(Calendar.DAY_OF_MONTH, -8);
        long startTimestamp = weekStart.getTime().getTime();
        String selection = ESM_Provider.ESM_Data.TIMESTAMP + ">= ?"
                + " AND " + ESM_Provider.ESM_Data.TIMESTAMP + "<= ?";
        String[] selectionArgs = new String[] {String.valueOf(startTimestamp), String.valueOf(endTimestamp)};
        Cursor localData = getContentResolver().query(ESM_Provider.ESM_Data.CONTENT_URI, null,
                selection, selectionArgs, ESM_Provider.ESM_Data.TIMESTAMP + " ASC");
        JSONArray weekData = new JSONArray();
        if (localData != null) {
            while (localData.moveToNext()) {
                // Retrieve the values from the cursor
                JSONObject jsonData = new JSONObject();
                for (String column : localData.getColumnNames()) {
                    int columnIndex = localData.getColumnIndex(column);
                    try {
                        if (column.toLowerCase(Locale.ROOT).contains("timestamp")) {
                            jsonData.put(column, localData.getLong(columnIndex));
                        } else {
                            switch (localData.getType(columnIndex)) {
                                case Cursor.FIELD_TYPE_INTEGER:
                                    jsonData.put(column, localData.getInt(columnIndex));
                                    break;
                                case Cursor.FIELD_TYPE_FLOAT:
                                    jsonData.put(column, localData.getFloat(columnIndex));
                                    break;
                                default:
                                    jsonData.put(column, localData.getString(columnIndex));
                                    break;
                            }
                        }
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }
                }
                weekData.put(jsonData);
            }
            localData.close();
        }
        return weekData;
    }

    private JSONArray getDummyESMData() {
        JSONArray dummyESMs = new JSONArray();
        String device_id = Aware.getSetting(getApplicationContext(), Aware_Preferences.DEVICE_ID);
        Cursor study = Aware.getStudy(getApplicationContext(),
                Aware.getSetting(getApplicationContext(), Aware_Preferences.WEBSERVICE_SERVER));
        if (study != null && study.moveToFirst()) {
            try {
                JSONObject localConfig = new JSONObject(study.getString(
                        study.getColumnIndex(Aware_Provider.Aware_Studies.STUDY_CONFIG)));
                JSONArray schedules = new JSONArray(localConfig.getString("schedules"));
                JSONArray questions = new JSONArray(localConfig.getString("questions"));

                HashMap<String, JSONObject> esm_questions = new HashMap<>();
                for (int i = 0; i < questions.length(); i++) {
                    try {
                        JSONObject questionJson = questions.getJSONObject(i);
                        String questionId = questionJson.getString("id");
                        esm_questions.put(questionId, new JSONObject().put("esm", questionJson));
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }
                }

                Calendar day = dayBefore;
                day.add(Calendar.DAY_OF_MONTH, -8);
                for (int t = 0; t < 7; t++) {
                    long curTimestamp = day.getTime().getTime();
                    // Set ESM schedules
                    for (int i = 0; i < schedules.length(); i++) {
                        try {
                            JSONObject scheduleJson = schedules.getJSONObject(i);
                            JSONArray questionIds = scheduleJson.getJSONArray("questions");
                            for (int j = 0; j < questionIds.length(); j++) {
                                JSONObject curQuestion = new JSONObject(esm_questions.get(questionIds.getString(j)).getString("esm"));
                                curQuestion.put("esm_trigger", scheduleJson.getString("title"));
                                JSONObject dummyESMResponse = new JSONObject();
                                dummyESMResponse.put(ESM_Provider.ESM_Data._ID, j);
                                dummyESMResponse.put(ESM_Provider.ESM_Data.TIMESTAMP, curTimestamp);
                                dummyESMResponse.put(ESM_Provider.ESM_Data.DEVICE_ID, device_id);
                                dummyESMResponse.put(ESM_Provider.ESM_Data.JSON, curQuestion.toString());
                                dummyESMResponse.put(ESM_Provider.ESM_Data.STATUS, 2);
                                dummyESMResponse.put(ESM_Provider.ESM_Data.EXPIRATION_THRESHOLD, 0);
                                dummyESMResponse.put(ESM_Provider.ESM_Data.NOTIFICATION_TIMEOUT, 0);
                                dummyESMResponse.put(ESM_Provider.ESM_Data.ANSWER_TIMESTAMP, curTimestamp);
                                if (curQuestion.getInt("id") == 3) {
//                                    dummyESMResponse.put(ESM_Provider.ESM_Data.ANSWER, (Math.round(Math.random() * (23 - 20)) + 20) + ":" + (Math.round(Math.random() * (59 - 10)) + 10));
                                    dummyESMResponse.put(ESM_Provider.ESM_Data.ANSWER, (Math.round(Math.random() * 10)) + ":" + (Math.round(Math.random() * (59 - 10)) + 10));
                                } else if (curQuestion.getInt("id") == 4) {
                                    dummyESMResponse.put(ESM_Provider.ESM_Data.ANSWER, (Math.round(Math.random() * (11 - 6)) + 6) + ":" + (Math.round(Math.random() * (59 - 10)) + 10));
                                } else {
                                    dummyESMResponse.put(ESM_Provider.ESM_Data.ANSWER, (Math.random() * (5 - 1)) + 1);
                                }
                                dummyESMResponse.put(ESM_Provider.ESM_Data.TRIGGER, scheduleJson.getString("title"));
                                dummyESMs.put(dummyESMResponse);
                            }
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }
                    day.add(Calendar.DAY_OF_MONTH, 1);
                }
            } catch (JSONException e) {
                e.printStackTrace();
            }
            study.close();
        }
        return dummyESMs;
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String s) {
        // Do nothing
    }

    private class queryWeekDataFromCloudDB extends AsyncTask<String, Void, Boolean> {
        private ProgressDialog mLoader;
        private JSONArray queryResults;
        @Override
        protected void onPreExecute() {
            mLoader = new ProgressDialog(Data.this);
            mLoader.setTitle(R.string.loading_data_title);
            mLoader.setMessage(getResources().getString(R.string.loading_join_study_msg));
            mLoader.setCancelable(false);
            mLoader.setIndeterminate(true);
            mLoader.show();
        }
        @Override
        protected Boolean doInBackground(String... strings) {
            long endTimestamp = dayBefore.getTime().getTime();
            Calendar weekStart = dayBefore;
            weekStart.add(Calendar.DAY_OF_MONTH, -7);
            long startTimestamp = weekStart.getTime().getTime();
            queryResults = Jdbc.queryWeekData(getApplicationContext(), "applications_foreground", startTimestamp, endTimestamp);
            if (queryResults != null) {
                return true;
            }
            return false;
        }

        @Override
        protected void onPostExecute(Boolean queryState) {
            mLoader.dismiss();

        }
    }
}
