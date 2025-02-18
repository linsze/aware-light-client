package com.aware.phone.ui.prefs;

import android.content.Context;
import android.preference.Preference;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;
import com.aware.phone.R;

public class ConfigSyncPref extends Preference {

    public ConfigSyncPref(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public ConfigSyncPref(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public ConfigSyncPref(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public ConfigSyncPref(Context context) {
        super(context);
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
