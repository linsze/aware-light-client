package com.aware.ui.esms;

import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.TextView;
import androidx.annotation.NonNull;
import com.aware.Aware;
import com.aware.ESM;
import com.aware.R;
import com.aware.providers.ESM_Provider;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * Created by NBerkel on 23/08/16.
 */
public class ESM_Number extends ESM_Question {
    private static EditText numberInput;

    public ESM_Number() throws JSONException {
        this.setType(ESM.TYPE_ESM_NUMBER);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.esm_number, container, false);
    }

    @Override
    public void onResume() {
        super.onResume();
        numberInput = getView().findViewById(R.id.esm_feedback);
        numberInput.requestFocus();
        InputMethodManager imm = (InputMethodManager) requireActivity().getSystemService(Context.INPUT_METHOD_SERVICE);
        imm.showSoftInput(numberInput, InputMethodManager.SHOW_FORCED);
    }

    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        try {
            TextView esm_title = (TextView) view.findViewById(R.id.esm_title);
            esm_title.setText(getTitle());
            esm_title.setMovementMethod(ScrollingMovementMethod.getInstance());

            TextView esm_instructions = (TextView) view.findViewById(R.id.esm_instructions);
            esm_instructions.setText(getInstructions());
            esm_instructions.setMovementMethod(ScrollingMovementMethod.getInstance());

            numberInput = (EditText) view.findViewById(R.id.esm_feedback);
//            numberInput.requestFocus();

            String savedNumber = (String) sharedViewModel.getStoredData(getID());
            if (savedNumber != null) {
                numberInput.setText(savedNumber);
            }

            numberInput.setOnClickListener(new View.OnClickListener() {
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
//            getActivity().getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_VISIBLE);
//            numberInput.post(() -> {
//                InputMethodManager imm = (InputMethodManager) getActivity().getSystemService(Context.INPUT_METHOD_SERVICE);
//                if (imm != null) {
//                    imm.showSoftInput(numberInput, InputMethodManager.SHOW_IMPLICIT);
//                }
//            });
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void saveData() {
        sharedViewModel.storeData(getID(), numberInput.getText().toString());

        ContentValues rowData = new ContentValues();
        rowData.put(ESM_Provider.ESM_Data.ANSWER_TIMESTAMP, System.currentTimeMillis());
        rowData.put(ESM_Provider.ESM_Data.ANSWER, numberInput.getText().toString());
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
