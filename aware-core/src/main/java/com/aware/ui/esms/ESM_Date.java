package com.aware.ui.esms;

import android.content.ContentValues;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CalendarView;
import android.widget.DatePicker;
import android.widget.TextView;
import androidx.annotation.NonNull;
import com.aware.Aware;
import com.aware.ESM;
import com.aware.R;
import com.aware.providers.ESM_Provider;
import org.json.JSONException;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.Calendar;

/**
 * Created by denzil on 01/11/2016.
 */

public class ESM_Date extends ESM_Question {

    public static final String esm_calendar = "esm_calendar";

    private static Calendar datePicked = null;

    public ESM_Date() throws JSONException {
        this.setType(ESM.TYPE_ESM_DATE);
    }

    public boolean isCalendar() throws JSONException {
        if (!this.esm.has(esm_calendar)) this.esm.put(esm_calendar, false);
        return this.esm.getBoolean(esm_calendar);
    }

    public ESM_Date setCalendar(boolean isCalendar) throws JSONException {
        this.esm.put(esm_calendar, isCalendar);
        return this;
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.esm_date, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        // TODO: Include SharedViewModel to restore answer from previous navigation
        datePicked = Calendar.getInstance();
        try {
            TextView esm_title = (TextView) view.findViewById(R.id.esm_title);
            esm_title.setText(getTitle());
            esm_title.setMovementMethod(ScrollingMovementMethod.getInstance());

            TextView esm_instructions = (TextView) view.findViewById(R.id.esm_instructions);
            esm_instructions.setText(getInstructions());
            esm_instructions.setMovementMethod(ScrollingMovementMethod.getInstance());

            final CalendarView calendarPicker = view.findViewById(R.id.esm_calendar);
            final DatePicker datePicker = view.findViewById(R.id.esm_datePicker);

            if (isCalendar() || Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) { //date picker doesn't exist for < 21
                calendarPicker.setVisibility(View.VISIBLE);
                calendarPicker.setDate(datePicked.getTimeInMillis());
                calendarPicker.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        try {
                            if (getExpirationThreshold() > 0 && expire_monitor != null)
                                expire_monitor.cancel(true);
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }
                });
                calendarPicker.setOnDateChangeListener(new CalendarView.OnDateChangeListener() {
                    @Override
                    public void onSelectedDayChange(@NonNull CalendarView calendarView, int year, int month, int dayOfMonth) {
                        datePicked.set(Calendar.YEAR, year);
                        datePicked.set(Calendar.MONTH, month);
                        datePicked.set(Calendar.DAY_OF_MONTH, dayOfMonth);
                    }
                });
                datePicker.setVisibility(View.GONE);
            } else {
                datePicker.setVisibility(View.VISIBLE);
                datePicker.init(datePicked.get(Calendar.YEAR), datePicked.get(Calendar.MONTH), datePicked.get(Calendar.DAY_OF_MONTH), new DatePicker.OnDateChangedListener() {
                    @Override
                    public void onDateChanged(DatePicker datePicker, int year, int month, int dayOfMonth) {
                        datePicked.set(Calendar.YEAR, year);
                        datePicked.set(Calendar.MONTH, month);
                        datePicked.set(Calendar.DAY_OF_MONTH, dayOfMonth);
                    }
                });
                calendarPicker.setVisibility(View.GONE);
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void saveData() {
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd Z");
        sharedViewModel.storeData(getID(), dateFormat.format(datePicked.getTime()));
        try {
            if (getExpirationThreshold() > 0 && expire_monitor != null)
                expire_monitor.cancel(true);

            ContentValues rowData = new ContentValues();
            rowData.put(ESM_Provider.ESM_Data.ANSWER_TIMESTAMP, System.currentTimeMillis());
            rowData.put(ESM_Provider.ESM_Data.ANSWER, dateFormat.format(datePicked.getTime()));
            rowData.put(ESM_Provider.ESM_Data.STATUS, ESM.STATUS_ANSWERED);

            getActivity().getContentResolver().update(ESM_Provider.ESM_Data.CONTENT_URI, rowData, ESM_Provider.ESM_Data._ID + "=" + getID(), null);

            Intent answer = new Intent(ESM.ACTION_AWARE_ESM_ANSWERED);
            JSONObject esmJSON = getEsm();
            esmJSON = esmJSON.put(ESM_Provider.ESM_Data._ID, getID());
            answer.putExtra(ESM.EXTRA_ESM, esmJSON.toString());
            answer.putExtra(ESM.EXTRA_ANSWER, rowData.getAsString(ESM_Provider.ESM_Data.ANSWER));
            getActivity().sendBroadcast(answer);

            if (Aware.DEBUG) Log.d(Aware.TAG, "Answer:" + rowData.toString());
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }
}
