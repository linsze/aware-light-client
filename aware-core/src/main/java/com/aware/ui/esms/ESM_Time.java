package com.aware.ui.esms;

import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.text.format.DateFormat;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import android.widget.TimePicker;
import androidx.annotation.NonNull;

import com.aware.Aware;
import com.aware.ESM;
import com.aware.R;
import com.aware.providers.ESM_Provider;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.Calendar;

public class ESM_Time extends ESM_Question {

    private static TimePicker timePicker;

    public ESM_Time() throws JSONException {
        this.setType(ESM.TYPE_ESM_TIME);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.esm_time, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        // Observe changes on ViewModel and reflect them on TimePicker
        sharedViewModel.getStoredData(getID()).observe(getViewLifecycleOwner(), value -> {
            if (value != null) {
                String savedTime = (String) value;
                String[] timeParts = savedTime.split(":");
                int savedHour = Integer.parseInt(timeParts[0]);
                int savedMinute = Integer.parseInt(timeParts[1]);
                timePicker.setHour(savedHour);
                timePicker.setMinute(savedMinute);
            }
        });
        try {
            TextView esm_title = (TextView) view.findViewById(R.id.esm_title);
            esm_title.setText(getTitle());
            esm_title.setMovementMethod(ScrollingMovementMethod.getInstance());

            TextView esm_instructions = (TextView) view.findViewById(R.id.esm_instructions);
            esm_instructions.setText(getInstructions());
            esm_instructions.setMovementMethod(ScrollingMovementMethod.getInstance());

            timePicker = (TimePicker) view.findViewById(R.id.timePicker);
            timePicker.setIs24HourView(DateFormat.is24HourFormat(getContext())); //makes the clock adjust to device's locale settings

            final Calendar chour = Calendar.getInstance();
            int initialHour = chour.get(Calendar.HOUR_OF_DAY);
            int initialMinute = chour.get(Calendar.MINUTE);
            timePicker.setHour(initialHour);
            timePicker.setMinute(initialMinute);
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }


    @Override
    public void saveData() {
        int selectedHour = timePicker.getHour();
        int selectedMinute = timePicker.getMinute();

        String timeToSave = String.format("%02d:%02d", selectedHour, selectedMinute);
        sharedViewModel.storeData(getID(), timeToSave);

        ContentValues rowData = new ContentValues();
        rowData.put(ESM_Provider.ESM_Data.ANSWER_TIMESTAMP, System.currentTimeMillis());
        rowData.put(ESM_Provider.ESM_Data.ANSWER, selectedHour + ":" + selectedMinute);
        rowData.put(ESM_Provider.ESM_Data.STATUS, ESM.STATUS_ANSWERED);

        getActivity().getContentResolver().update(ESM_Provider.ESM_Data.CONTENT_URI, rowData, ESM_Provider.ESM_Data._ID + "=" + getID(), null);

        Intent answer = new Intent(ESM.ACTION_AWARE_ESM_ANSWERED);
        JSONObject esmJSON = getEsm();
        try {
            esmJSON = esmJSON.put(ESM_Provider.ESM_Data._ID, getID());
        } catch (JSONException e) {
            throw new RuntimeException(e);
        }
        answer.putExtra(ESM.EXTRA_ESM, esmJSON.toString());
        answer.putExtra(ESM.EXTRA_ANSWER, rowData.getAsString(ESM_Provider.ESM_Data.ANSWER));
        getActivity().sendBroadcast(answer);

        if (Aware.DEBUG) Log.d(Aware.TAG, "Answer:" + rowData.toString());
    }
}
