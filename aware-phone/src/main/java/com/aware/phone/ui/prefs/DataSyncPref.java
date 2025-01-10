package com.aware.phone.ui.prefs;

import android.content.Context;
import android.content.Intent;
import android.preference.Preference;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import com.aware.Aware;
import com.aware.phone.R;

public class DataSyncPref extends Preference {
    public DataSyncPref(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public DataSyncPref(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public DataSyncPref(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public DataSyncPref(Context context) {
        super(context);
    }

    @Override
    protected View onCreateView(ViewGroup parent) {
        super.onCreateView(parent);
        LayoutInflater inflater = (LayoutInflater) getContext().getSystemService(
                Context.LAYOUT_INFLATER_SERVICE);
        View view = inflater.inflate(R.layout.pref_data_sync, parent, false);

        view.findViewById(R.id.btn_sync_data_config).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Toast.makeText(getContext(), "Syncing data...", Toast.LENGTH_SHORT).show();
                Intent sync = new Intent(Aware.ACTION_AWARE_SYNC_DATA);
                getContext().sendBroadcast(sync);
            }
        });

        return view;
    }
}
