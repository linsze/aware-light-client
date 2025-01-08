package com.aware.ui.esms;

import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.annotation.NonNull;

import com.aware.Aware;
import com.aware.ESM;
import com.aware.R;
import com.aware.providers.ESM_Provider;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

/**
 * Created by denzilferreira on 21/02/16.
 */
public class ESM_QuickAnswer extends ESM_Question {

    public static final String esm_quick_answers = "esm_quick_answers";

    private static String selected_answer;

    private static ArrayList<Button> answer_buttons = new ArrayList<>();

    public ESM_QuickAnswer() throws JSONException {
        this.setType(ESM.TYPE_ESM_QUICK_ANSWERS);
    }

    public JSONArray getQuickAnswers() throws JSONException {
        if (!this.esm.has(esm_quick_answers)) {
            this.esm.put(esm_quick_answers, new JSONArray());
        }
        return this.esm.getJSONArray(esm_quick_answers);
    }

    public ESM_QuickAnswer setQuickAnswers(JSONArray quickAnswers) throws JSONException {
        this.esm.put(this.esm_quick_answers, quickAnswers);
        return this;
    }

    public ESM_QuickAnswer addQuickAnswer(String answer) throws JSONException {
        JSONArray quicks = getQuickAnswers();
        quicks.put(answer);
        this.setQuickAnswers(quicks);
        return this;
    }

    public ESM_QuickAnswer removeQuickAnswer(String answer) throws JSONException {
        JSONArray quick = getQuickAnswers();
        JSONArray newQuick = new JSONArray();
        for (int i = 0; i < quick.length(); i++) {
            if (quick.getString(i).equals(answer)) continue;
            newQuick.put(quick.getString(i));
        }
        this.setQuickAnswers(newQuick);
        return this;
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.esm_quick, container, false);
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

            final JSONArray answers = getQuickAnswers();
            final LinearLayout answersHolder = (LinearLayout) view.findViewById(R.id.esm_answers);

            //If we have more than 3 possibilities, use a vertical layout for UX
            if (answers.length() > 3) {
                answersHolder.setOrientation(LinearLayout.VERTICAL);
            }

            String savedAnswer = (String) sharedViewModel.getStoredData(getID());
            if (savedAnswer != null) {
                selected_answer = savedAnswer;
            }

            for (int i = 0; i < answers.length(); i++) {
                Button answer = new Button(getActivity());
                LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.WRAP_CONTENT, 1.0f);
                //Fixed: buttons now of the same height regardless of content.
                params.height = WindowManager.LayoutParams.MATCH_PARENT;
                answer.setLayoutParams(params);
                answer.setText(answers.getString(i));
                answer.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        try {
                            if (getExpirationThreshold() > 0 && expire_monitor != null)
                                expire_monitor.cancel(true);

                            selected_answer = (String) answer.getText();
                            for (Button btn : answer_buttons) {
                                btn.setBackgroundColor(Color.LTGRAY);
                                btn.setTextColor(Color.BLACK);
                            }
                            // Highlight the current selection
                            answer.setBackgroundColor(R.color.primary);
                            answer.setTextColor(Color.WHITE);

                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }
                });
                answersHolder.addView(answer);
                answer_buttons.add(answer);
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void saveData() {
        sharedViewModel.storeData(getID(), selected_answer);

        ContentValues rowData = new ContentValues();
        rowData.put(ESM_Provider.ESM_Data.ANSWER_TIMESTAMP, System.currentTimeMillis());
        rowData.put(ESM_Provider.ESM_Data.STATUS, ESM.STATUS_ANSWERED);
        rowData.put(ESM_Provider.ESM_Data.ANSWER, selected_answer);

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
