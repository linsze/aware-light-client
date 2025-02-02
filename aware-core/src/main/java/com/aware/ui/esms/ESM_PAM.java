package com.aware.ui.esms;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.gridlayout.widget.GridLayout;
import com.aware.Aware;
import com.aware.ESM;
import com.aware.R;
import com.aware.providers.ESM_Provider;
import com.koushikdutta.ion.Ion;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.Random;

/**
 * Created by denzil on 02/11/2016.
 * Based on JP at Cornell's work on Photographic Affect Meter (PAM):
 * https://github.com/ohmage/ohmage-pam
 */
public class ESM_PAM extends ESM_Question {

    private String pam_selected = "";

    public static final String esm_pam = "esm_pam";

    public ESM_PAM() throws JSONException {
        this.setType(ESM.TYPE_ESM_PAM);
    }

    /**
     * Get PAM JSONArray with picture URLs
     *
     * @return
     * @throws JSONException
     */
    public JSONArray getPAM() throws JSONException {
        if (!this.esm.has(esm_pam)) {
            this.esm.put(esm_pam, new JSONArray());
        }
        return this.esm.getJSONArray(esm_pam);
    }

    /**
     * Set PAM list of picture URLs
     *
     * @param pam
     * @return
     * @throws JSONException
     */
    public ESM_PAM setPAM(JSONArray pam) throws JSONException {
        this.esm.put(esm_pam, pam);
        return this;
    }

    private String moodDescription(int mood) {
        switch (mood) {
            case 1:
                return "afraid";
            case 2:
                return "tense";
            case 3:
                return "excited";
            case 4:
                return "delighted";
            case 5:
                return "frustrated";
            case 6:
                return "angry";
            case 7:
                return "happy";
            case 8:
                return "glad";
            case 9:
                return "miserable";
            case 10:
                return "sad";
            case 11:
                return "calm";
            case 12:
                return "satisfied";
            case 13:
                return "gloomy";
            case 14:
                return "tired";
            case 15:
                return "sleepy";
            case 16:
                return "serene";
            default:
                return "-";
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.esm_pam, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        pam_selected = "";

        String savedPam = (String) sharedViewModel.getStoredData(getID());
        if (savedPam != null) {
            pam_selected = savedPam;
            //TODO: Reflect restored data on display
        }
        try {
            TextView esm_title = (TextView) view.findViewById(R.id.esm_title);
            esm_title.setText(getTitle());
            esm_title.setMovementMethod(ScrollingMovementMethod.getInstance());

            TextView esm_instructions = (TextView) view.findViewById(R.id.esm_instructions);
            esm_instructions.setText(getInstructions());
            esm_instructions.setMovementMethod(ScrollingMovementMethod.getInstance());

            final GridLayout answersHolder = (GridLayout) view.findViewById(R.id.esm_pam);
            answersHolder.setOnClickListener(new View.OnClickListener() {
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

            JSONArray moods = getPAM(); //0-indexed
            if (moods.length() == 0) {
                //Load by default ours
                moods.put("http://awareframework.com/public/pam/afraid");
                moods.put("http://awareframework.com/public/pam/tense");
                moods.put("http://awareframework.com/public/pam/excited");
                moods.put("http://awareframework.com/public/pam/delighted");
                moods.put("http://awareframework.com/public/pam/frustrated");
                moods.put("http://awareframework.com/public/pam/angry");
                moods.put("http://awareframework.com/public/pam/happy");
                moods.put("http://awareframework.com/public/pam/glad");
                moods.put("http://awareframework.com/public/pam/miserable");
                moods.put("http://awareframework.com/public/pam/sad");
                moods.put("http://awareframework.com/public/pam/calm");
                moods.put("http://awareframework.com/public/pam/satisfied");
                moods.put("http://awareframework.com/public/pam/gloomy");
                moods.put("http://awareframework.com/public/pam/tired");
                moods.put("http://awareframework.com/public/pam/sleepy");
                moods.put("http://awareframework.com/public/pam/serene");
            }

            for (int i = 1; i < 17; i++) {

                final int childPos = i;

                ImageView moodOption = (ImageView) view.findViewById(getResources().getIdentifier("pos" + i, "id", getActivity().getPackageName()));

                String mood_picture_url = moods.getString(i-1);

                Random rand_pic = new Random(System.currentTimeMillis());
                Integer pic = 1 + rand_pic.nextInt(3);

                //Asynchronously download mood image and caches automatically
                Ion.getDefault(getActivity().getApplicationContext()).getConscryptMiddleware().enable(false);
                Ion.with(moodOption).placeholder(R.drawable.square).load(mood_picture_url + "/" + pic + ".jpg");

                moodOption.setTag(moodDescription(i));
                moodOption.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        try {
                            if (getExpirationThreshold() > 0 && expire_monitor != null)
                                expire_monitor.cancel(true);
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                        pam_selected = view.getTag().toString();

                        answersHolder.getChildAt(childPos-1).setSelected(true);

                        for (int j=1; j<17; j++){
                            if (childPos == j) {
                                continue;
                            } else
                                answersHolder.getChildAt(j-1).setSelected(false);
                        }
                    }
                });
            }

        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void saveData() {
        sharedViewModel.storeData(getID(), pam_selected);

        ContentValues rowData = new ContentValues();
        rowData.put(ESM_Provider.ESM_Data.ANSWER_TIMESTAMP, System.currentTimeMillis());
        rowData.put(ESM_Provider.ESM_Data.ANSWER, pam_selected);
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
