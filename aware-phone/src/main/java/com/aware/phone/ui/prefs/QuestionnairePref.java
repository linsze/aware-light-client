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

import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.ESM;
import com.aware.phone.R;
import com.aware.providers.ESM_Provider;
import com.aware.ui.ESM_Queue;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Locale;

public class QuestionnairePref extends Preference {
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

    private boolean setupQuestionnaire(String title) {
        String esmSchedule = Aware.getSetting(getContext(), Aware_Preferences.ESM_SCHEDULES);
        JSONObject esmScheduleJson = new JSONObject();
        JSONArray esmArray = new JSONArray();
        long esm_timestamp = System.currentTimeMillis();
        boolean queueNotAnswered = false;

        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd", Locale.getDefault());
        Calendar calendar = Calendar.getInstance();
        String questionnaireDate = dateFormat.format(calendar.getTime());

        try {
            if (!esmSchedule.equals("")) {
                esmScheduleJson = new JSONObject(esmSchedule);
            }
            if (esmScheduleJson.has(title)) {
                esmArray = new JSONArray(esmScheduleJson.getString(title));
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
        return queueNotAnswered;
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
                if (setupQuestionnaire("Daily morning schedule")) {
                    Intent intent_ESM = new Intent(getContext(), ESM_Queue.class);
                    intent_ESM.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                    getContext().startActivity(intent_ESM);
                } else {
                    Toast.makeText(getContext(), "Questionnaire has been answered", Toast.LENGTH_LONG).show();
                }
            }
        });

        view.findViewById(R.id.btn_evening_questionnaire).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (setupQuestionnaire("Daily evening schedule")) {
                    Intent intent_ESM = new Intent(getContext(), ESM_Queue.class);
                    intent_ESM.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                    getContext().startActivity(intent_ESM);
                } else {
                    Toast.makeText(getContext(), "Questionnaire has been answered", Toast.LENGTH_LONG).show();
                }
            }
        });

        return view;
    }
}
