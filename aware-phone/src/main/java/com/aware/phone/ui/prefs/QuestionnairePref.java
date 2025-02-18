package com.aware.phone.ui.prefs;

import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.preference.Preference;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;
import android.app.AlertDialog;

import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.ESM;
import com.aware.phone.R;
import com.aware.providers.ESM_Provider;
import com.aware.ui.ESM_Queue;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.Locale;

public class QuestionnairePref extends Preference {
    private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd", Locale.getDefault());

    public QuestionnairePref(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public QuestionnairePref(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public QuestionnairePref(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public QuestionnairePref(Context context) {
        super(context);
    }

    private String retrieveESMDateToAnswer(String title) {
        // Get the date corresponding to the latest ESM that has been submitted.
        String esmDate = "";
        String esmTrigger = "";
        // Doesn't prompt answering if ESM for the day has been submitted
        Cursor answeredEsm = getContext().getContentResolver().query(ESM_Provider.ESM_Data.CONTENT_URI,null,
                ESM_Provider.ESM_Data.TRIGGER + "='" + title+ "' AND " + ESM_Provider.ESM_Data.STATUS + '=' + ESM.STATUS_SUBMITTED,
                null, ESM_Provider.ESM_Data.TIMESTAMP + " DESC LIMIT 1");
        if (answeredEsm != null && answeredEsm.moveToFirst()) {
            esmDate = answeredEsm.getString(answeredEsm.getColumnIndex(ESM_Provider.ESM_Data.DATE));
            esmTrigger = answeredEsm.getString(answeredEsm.getColumnIndex(ESM_Provider.ESM_Data.TRIGGER));
        }
        if (answeredEsm != null && !answeredEsm.isClosed()) answeredEsm.close();

        // Get the date for the next day after the latest submitted ESM.
        Calendar esmDay = Calendar.getInstance();
        if (!esmDate.equals("")) {
            try {
                Date date = dateFormat.parse(esmDate);
                esmDay.setTime(date);
                esmDay.add(Calendar.DAY_OF_MONTH, 1);

            } catch (ParseException e) {
                e.printStackTrace();
            }
        }

        // Make sure that the date is valid (i.e. before tomorrow)
        Calendar tomorrow = Calendar.getInstance();
        tomorrow.add(Calendar.DAY_OF_MONTH, 1);
        tomorrow.set(Calendar.HOUR_OF_DAY, 0);
        tomorrow.set(Calendar.MINUTE, 0);
        tomorrow.set(Calendar.SECOND, 0);
        tomorrow.set(Calendar.MILLISECOND, 0);

        if (esmDay.before(tomorrow)) {
            return dateFormat.format(esmDay.getTime());
        }
        return "";
    }

    private void showDialog(String esmDate, String questionnaireTitle) {
        String esmDateToDisplay = esmDate;
        try {
            Date currentEsmDate = dateFormat.parse(esmDate);
            SimpleDateFormat dayFormat = new SimpleDateFormat("EEEE", Locale.getDefault());
            String dayOfWeek = dayFormat.format(currentEsmDate);
            esmDateToDisplay += " (" + dayOfWeek + ")";
        } catch (ParseException e) {
            e.printStackTrace();
        }

        AlertDialog.Builder builder = new AlertDialog.Builder(getContext());
        builder.setTitle("Answer Questionnaire")
                .setMessage("Answer questionnaire for " + esmDateToDisplay + "?")
                .setPositiveButton("Yes", (dialog, which) -> {
                    dialog.dismiss();
                    setupQuestionnaire(esmDate, questionnaireTitle);
                })
                .setNeutralButton("No", (dialog, which) -> {
                    dialog.dismiss();
                });

        AlertDialog dialog = builder.create();
        dialog.show();
    }

    private void setupQuestionnaire(String questionnaireDate, String questionnaireTitle) {
        String esmSchedule = Aware.getSetting(getContext(), Aware_Preferences.ESM_SCHEDULES);
        JSONObject esmScheduleJson = new JSONObject();
        JSONArray esmArray = new JSONArray();
        long esm_timestamp = System.currentTimeMillis();
        boolean queueNotAnswered = false;

//        Calendar calendar = Calendar.getInstance();
//        String questionnaireDate = dateFormat.format(calendar.getTime());

        try {
            if (!esmSchedule.equals("")) {
                esmScheduleJson = new JSONObject(esmSchedule);
            }
            if (esmScheduleJson.has(questionnaireTitle)) {
                esmArray = new JSONArray(esmScheduleJson.getString(questionnaireTitle));
            }

            for (int i = 0; i < esmArray.length(); i++) {
                JSONObject esm = esmArray.getJSONObject(i).getJSONObject(ESM.EXTRA_ESM);
                String esmString = esm.toString();
                boolean esmAnswered = false;

                // Doesn't prompt answering if ESM for the day has been submitted
                Cursor answeredEsm = getContext().getContentResolver().query(ESM_Provider.ESM_Data.CONTENT_URI,null,
                        ESM_Provider.ESM_Data.JSON + " LIKE ? AND " + ESM_Provider.ESM_Data.DATE + "='" + questionnaireDate + "'",
                        new String[]{"%" + esmString + "%"}, ESM_Provider.ESM_Data.TIMESTAMP + " DESC");
                int existingEsmId = -1;
                if (answeredEsm != null && answeredEsm.moveToFirst()) {
                    do {
                        String existingEsm = answeredEsm.getString(answeredEsm.getColumnIndex(ESM_Provider.ESM_Data.JSON));
                        if (existingEsm.equals(esmString)) {
                            existingEsmId = answeredEsm.getInt(answeredEsm.getColumnIndex(ESM_Provider.ESM_Data._ID));
                            // Consider states other than submitted ones to prevent duplicated questions
                            if (answeredEsm.getInt(answeredEsm.getColumnIndex(ESM_Provider.ESM_Data.STATUS)) == ESM.STATUS_SUBMITTED) {
                                esmAnswered = true;
                            }
                            break;
                        }
                    } while (answeredEsm.moveToNext());
                }
                if (answeredEsm != null && !answeredEsm.isClosed()) answeredEsm.close();

                ContentValues rowData = new ContentValues();
                rowData.put(ESM_Provider.ESM_Data.TIMESTAMP, esm_timestamp + i);
                rowData.put(ESM_Provider.ESM_Data.DEVICE_ID, Aware.getSetting(getContext(), Aware_Preferences.DEVICE_ID));
                rowData.put(ESM_Provider.ESM_Data.JSON, esm.toString());
                rowData.put(ESM_Provider.ESM_Data.EXPIRATION_THRESHOLD, esm.optInt(ESM_Provider.ESM_Data.EXPIRATION_THRESHOLD));
                rowData.put(ESM_Provider.ESM_Data.NOTIFICATION_TIMEOUT, esm.optInt(ESM_Provider.ESM_Data.NOTIFICATION_TIMEOUT));
                rowData.put(ESM_Provider.ESM_Data.STATUS, ESM.STATUS_NEW);
                rowData.put(ESM_Provider.ESM_Data.TRIGGER, esm.optString(ESM_Provider.ESM_Data.TRIGGER));
                rowData.put(ESM_Provider.ESM_Data.DATE, questionnaireDate);

                if (existingEsmId == -1) {
                    getContext().getContentResolver().insert(ESM_Provider.ESM_Data.CONTENT_URI, rowData);
                    queueNotAnswered = true;
                } else if (existingEsmId != -1 && !esmAnswered) {
                    // Allow updating if ESM hasn't been submitted
                    getContext().getContentResolver().update(ESM_Provider.ESM_Data.CONTENT_URI, rowData, ESM_Provider.ESM_Data._ID + "=" + existingEsmId, null);
                    queueNotAnswered = true;
                }
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }

        if (queueNotAnswered) {
            Intent intent_ESM = new Intent(getContext(), ESM_Queue.class);
            intent_ESM.putExtra(ESM.EXTRA_DATE, questionnaireDate);
            intent_ESM.putExtra(ESM.EXTRA_SCHEDULE, questionnaireTitle);
            intent_ESM.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            getContext().startActivity(intent_ESM);
        }
    }
    @Override
    protected View onCreateView(ViewGroup parent) {
        super.onCreateView(parent);
        LayoutInflater inflater = (LayoutInflater) getContext().getSystemService(
                Context.LAYOUT_INFLATER_SERVICE);
        View view = inflater.inflate(R.layout.pref_questionnaire, parent, false);

        view.findViewById(R.id.btn_day_questionnaire).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String scheduleTitle = "Daily morning schedule";
                String esmDateToPrompt = retrieveESMDateToAnswer(scheduleTitle);
                if (!esmDateToPrompt.equals("")) {
                    showDialog(esmDateToPrompt, scheduleTitle);
                } else {
                    Toast.makeText(getContext(), "Questionnaire has been answered", Toast.LENGTH_LONG).show();
                }
            }
        });

        view.findViewById(R.id.btn_evening_questionnaire).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String scheduleTitle = "Daily evening schedule";
                String esmDateToPrompt = retrieveESMDateToAnswer(scheduleTitle);
                if (!esmDateToPrompt.equals("")) {
                    showDialog(esmDateToPrompt, scheduleTitle);
                } else {
                    Toast.makeText(getContext(), "Questionnaire has been answered", Toast.LENGTH_LONG).show();
                }
            }
        });

        return view;
    }
}
