package com.aware.ui.esms;

import android.content.ContentValues;
import android.content.Intent;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RatingBar;
import android.widget.TextView;
import androidx.annotation.NonNull;
import com.aware.Aware;
import com.aware.ESM;
import com.aware.R;
import com.aware.providers.ESM_Provider;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * Created by denzilferreira on 21/02/16.
 */
public class ESM_Likert extends ESM_Question {

    public static final String esm_likert_max = "esm_likert_max";
    public static final String esm_likert_max_label = "esm_likert_max_label";
    public static final String esm_likert_min_label = "esm_likert_min_label";
    public static final String esm_likert_step = "esm_likert_step";

    private static RatingBar ratingBar;

    public ESM_Likert() throws JSONException {
        this.setType(ESM.TYPE_ESM_LIKERT);
    }

    public int getLikertMax() throws JSONException {
        if(!this.esm.has(esm_likert_max)) {
            this.esm.put(esm_likert_max, 5);
        }
        return this.esm.getInt(esm_likert_max);
    }

    public ESM_Likert setLikertMax(int max) throws JSONException {
        this.esm.put(esm_likert_max, max);
        return this;
    }

    public String getLikertMaxLabel() throws JSONException {
        if(!this.esm.has(esm_likert_max_label)) {
            this.esm.put(esm_likert_max_label, "");
        }
        return this.esm.getString(esm_likert_max_label);
    }

    public ESM_Likert setLikertMaxLabel(String label) throws JSONException {
        this.esm.put(esm_likert_max_label, label);
        return this;
    }

    public String getLikertMinLabel() throws JSONException {
        if(!this.esm.has(esm_likert_min_label)) {
            this.esm.put(esm_likert_min_label, "");
        }
        return this.esm.getString(esm_likert_min_label);
    }

    public ESM_Likert setLikertMinLabel(String label) throws JSONException {
        this.esm.put(esm_likert_min_label, label);
        return this;
    }

    public double getLikertStep() throws JSONException {
        if(!this.esm.has(esm_likert_step)) {
            this.esm.put(esm_likert_step, 1.0);
        }
        return this.esm.getDouble(esm_likert_step);
    }

    public ESM_Likert setLikertStep(double step) throws JSONException {
        this.esm.put(esm_likert_step, step);
        return this;
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.esm_likert, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        sharedViewModel.getStoredData(getID()).observe(getViewLifecycleOwner(), value -> {
            if (value != null) {
                Float savedRating = (Float) value;
                ratingBar.setRating(savedRating);
            }
        });

        try {
            TextView esm_title = (TextView) view.findViewById(R.id.esm_title);
            esm_title.setText(getTitle());
            esm_title.setMovementMethod(ScrollingMovementMethod.getInstance());

            TextView esm_instructions = (TextView) view.findViewById(R.id.esm_instructions);
            esm_instructions.setText(getInstructions());
            esm_instructions.setMovementMethod(ScrollingMovementMethod.getInstance());

            ratingBar = (RatingBar) view.findViewById(R.id.esm_likert);
            ratingBar.setNumStars(getLikertMax());
            ratingBar.setMax(getLikertMax());
            ratingBar.setStepSize((float) getLikertStep());

            ratingBar.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    try {
                        if (getExpirationThreshold() > 0 && expire_monitor != null)
                            expire_monitor.cancel(true);
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }
                }
            });

            TextView min_label = (TextView) view.findViewById(R.id.esm_min);
            min_label.setText(getLikertMinLabel());

            TextView max_label = (TextView) view.findViewById(R.id.esm_max);
            max_label.setText(getLikertMaxLabel());
        }catch (JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void saveData() {
        sharedViewModel.storeData(getID(), ratingBar.getRating());

        ContentValues rowData = new ContentValues();
        rowData.put(ESM_Provider.ESM_Data.ANSWER_TIMESTAMP, System.currentTimeMillis());
        rowData.put(ESM_Provider.ESM_Data.ANSWER, ratingBar.getRating());
        rowData.put(ESM_Provider.ESM_Data.STATUS, ESM.STATUS_ANSWERED);

        getContext().getContentResolver().update(ESM_Provider.ESM_Data.CONTENT_URI, rowData, ESM_Provider.ESM_Data._ID + "=" + getID(), null);

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
