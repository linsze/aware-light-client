package com.aware.ui.esms;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.text.format.DateFormat;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.DatePicker;
import android.widget.TextView;
import android.widget.TimePicker;
import androidx.annotation.NonNull;
import androidx.viewpager.widget.PagerAdapter;
import androidx.viewpager.widget.ViewPager;
import com.aware.Aware;
import com.aware.ESM;
import com.aware.R;
import com.aware.providers.ESM_Provider;
import com.google.android.material.tabs.TabLayout;
import org.json.JSONException;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.Calendar;

/**
 * Created by denzil on 01/11/2016.
 */

public class ESM_DateTime extends ESM_Question {

    private static Calendar datePicked = null;

    public ESM_DateTime() throws JSONException {
        this.setType(ESM.TYPE_ESM_DATETIME);
    }

    public class DateTimePagerAdapter extends PagerAdapter {
        private Context mContext;

        public DateTimePagerAdapter(Context context) {
            mContext = context;
        }

        @Override
        public Object instantiateItem(ViewGroup container, int position) {
            LayoutInflater inflater = LayoutInflater.from(mContext);
            ViewGroup layout = null;
            switch (position) {
                case 0:
                    layout = (ViewGroup) inflater.inflate(R.layout.esm_datetime_date, container, false);

                    final DatePicker datePicker = (DatePicker) layout.findViewById(R.id.datePicker);

                    final Calendar cdate = Calendar.getInstance();
                    int year = cdate.get(Calendar.YEAR);
                    int month = cdate.get(Calendar.MONTH);
                    int day = cdate.get(Calendar.DAY_OF_MONTH);

                    datePicker.init(year, month, day, new DatePicker.OnDateChangedListener() {
                        @Override
                        public void onDateChanged(DatePicker datePicker, int year, int month, int day) {
                            datePicked.set(Calendar.DAY_OF_MONTH, day);
                            datePicked.set(Calendar.MONTH, month);
                            datePicked.set(Calendar.YEAR, year);
                        }
                    });

                    container.addView(layout);
                    break;
                case 1:
                    layout = (ViewGroup) inflater.inflate(R.layout.esm_datetime_time, container, false);

                    final TimePicker timePicker = (TimePicker) layout.findViewById(R.id.timePicker);
                    timePicker.setIs24HourView(DateFormat.is24HourFormat(getContext())); //makes the clock adjust to device's locale settings

                    final Calendar chour = Calendar.getInstance();
                    int hour = chour.get(Calendar.HOUR_OF_DAY);
                    int minute = chour.get(Calendar.MINUTE);
                    if (Build.VERSION.SDK_INT >=23) {
                        timePicker.setHour(hour);
                    } else {
                        timePicker.setCurrentHour(hour);
                    }
                    if (Build.VERSION.SDK_INT >= 23) {
                        timePicker.setMinute(minute);
                    } else {
                        timePicker.setCurrentMinute(minute);
                    }

                    timePicker.setOnTimeChangedListener(new TimePicker.OnTimeChangedListener() {
                        @Override
                        public void onTimeChanged(TimePicker timePicker, int hour, int minute) {
                            datePicked.set(Calendar.HOUR_OF_DAY, hour);
                            datePicked.set(Calendar.MINUTE, minute);
                        }
                    });

                    container.addView(layout);
                    break;
            }
            return layout;
        }

        @Override
        public void destroyItem(ViewGroup container, int position, Object view) {
            container.removeView((View) view);
        }

        @Override
        public CharSequence getPageTitle(int position) {
            String[] titles = {"Date", "Time"};
            return titles[position];
        }

        @Override
        public int getCount() {
            return 2;
        }

        @Override
        public boolean isViewFromObject(View view, Object object) {
            return view == object;
        }
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.esm_datetime, container, false);
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

            final ViewPager datetimePager = (ViewPager) view.findViewById(R.id.datetimepager);
            DateTimePagerAdapter dateTimePagerAdapter = new DateTimePagerAdapter(getContext());
            datetimePager.setAdapter(dateTimePagerAdapter);
            datetimePager.setOnClickListener(new View.OnClickListener() {
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

            final TabLayout tabLayout = view.findViewById(R.id.datetimetabs);
            tabLayout.setupWithViewPager(datetimePager, true);
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void saveData() {
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z");
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
