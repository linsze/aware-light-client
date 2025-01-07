package com.aware.ui.esms;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.*;
import androidx.annotation.NonNull;
import com.aware.Aware;
import com.aware.ESM;
import com.aware.R;
import com.aware.providers.ESM_Provider;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * Created by denzilferreira on 21/02/16.
 */
public class ESM_Radio extends ESM_Question {

    public static final String esm_radios = "esm_radios";
    private static RadioGroup radioOptions;

    public ESM_Radio() throws JSONException {
        this.setType(ESM.TYPE_ESM_RADIO);
    }

    public JSONArray getRadios() throws JSONException {
        if (!this.esm.has(esm_radios)) {
            this.esm.put(esm_radios, new JSONArray());
        }
        return this.esm.getJSONArray(esm_radios);
    }

    public ESM_Radio setRadios(JSONArray radios) throws JSONException {
        this.esm.put(esm_radios, radios);
        return this;
    }

    public ESM_Radio addRadio(String option) throws JSONException {
        JSONArray radios = getRadios();
        radios.put(option);
        this.setRadios(radios);
        return this;
    }

    public ESM_Radio removeRadio(String option) throws JSONException {
        JSONArray radios = getRadios();
        JSONArray newRadios = new JSONArray();
        for (int i = 0; i < radios.length(); i++) {
            if (radios.getString(i).equals(option)) continue;
            newRadios.put(radios.getString(i));
        }
        this.setRadios(newRadios);
        return this;
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.esm_radio, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        //TODO: Include SharedViewModel to restore answer from previous navigation
        try {
            TextView esm_title = (TextView) view.findViewById(R.id.esm_title);
            esm_title.setText(getTitle());
            esm_title.setMovementMethod(ScrollingMovementMethod.getInstance());

            TextView esm_instructions = (TextView) view.findViewById(R.id.esm_instructions);
            esm_instructions.setText(getInstructions());
            esm_instructions.setMovementMethod(ScrollingMovementMethod.getInstance());

            radioOptions = (RadioGroup) view.findViewById(R.id.esm_radio);
            radioOptions.setOnClickListener(new View.OnClickListener() {
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

            final JSONArray radios = getRadios();
            for (int i = 0; i < radios.length(); i++) {
                final RadioButton radioOption = new RadioButton(getActivity());
                radioOption.setId(i);
                radioOption.setText(radios.getString(i));
                radioOptions.addView(radioOption);

                if (radios.getString(i).equals(getResources().getString(R.string.aware_esm_other))) {
                    radioOption.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View v) {
                            final Dialog editOther = new Dialog(getActivity());
                            editOther.setTitle(getResources().getString(R.string.aware_esm_other_follow));
                            editOther.getWindow().setGravity(Gravity.TOP);
                            editOther.getWindow().setLayout(WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT);

                            LinearLayout editor = new LinearLayout(getActivity());
                            editor.setOrientation(LinearLayout.VERTICAL);

                            editOther.setContentView(editor);
                            editOther.show();

                            final EditText otherText = new EditText(getActivity());
                            otherText.setHint(getResources().getString(R.string.aware_esm_other_follow));
                            editor.addView(otherText);
                            otherText.requestFocus();
                            editOther.getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_VISIBLE);

                            Button confirm = new Button(getActivity());
                            confirm.setText("OK");
                            confirm.setOnClickListener(new View.OnClickListener() {
                                @Override
                                public void onClick(View v) {
                                    if (otherText.length() > 0)
                                        radioOption.setText(otherText.getText());
                                    editOther.dismiss();
                                }
                            });
                            editor.addView(confirm);
                        }
                    });
                }
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void saveData() {
        ContentValues rowData = new ContentValues();
        rowData.put(ESM_Provider.ESM_Data.ANSWER_TIMESTAMP, System.currentTimeMillis());

        if (radioOptions.getCheckedRadioButtonId() != -1) {
            RadioButton selected = (RadioButton) radioOptions.getChildAt(radioOptions.getCheckedRadioButtonId());
            String selectValue = String.valueOf(selected.getText()).trim();
            rowData.put(ESM_Provider.ESM_Data.ANSWER, selectValue);
            sharedViewModel.storeData(getID(), selectValue);
        }
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
