package com.aware.phone.ui.prefs;

import android.content.Context;
import android.content.ContextWrapper;
import android.preference.Preference;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import com.aware.Aware;
import com.aware.phone.R;

public class RefreshSamplingPref extends Preference {

    public RefreshSamplingPref(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public RefreshSamplingPref(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public RefreshSamplingPref(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public RefreshSamplingPref(Context context) {
        super(context);
    }
    @Override
    protected View onCreateView(ViewGroup parent) {
        super.onCreateView(parent);
        LayoutInflater inflater = (LayoutInflater) getContext().getSystemService(
                Context.LAYOUT_INFLATER_SERVICE);
        View view = inflater.inflate(R.layout.pref_refresh_sampling, parent, false);

        view.findViewById(R.id.btn_refresh_sampling).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Aware.startAWARE(getContext(), true);
                Toast.makeText(getContext(), "Data collection has been refreshed", Toast.LENGTH_SHORT).show();
            }
        });

        return view;
    }
}
