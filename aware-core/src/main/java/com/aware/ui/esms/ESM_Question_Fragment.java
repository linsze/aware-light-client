package com.aware.ui.esms;


import android.os.Bundle;
import androidx.fragment.app.Fragment;
import com.aware.ESM;
import com.aware.R;
import com.aware.providers.ESM_Provider;
import org.json.JSONException;
import org.json.JSONObject;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

public class ESM_Question_Fragment extends Fragment {
    /**
     * A Fragment holder to contain an ESM question.
     */
    private static final String ESM_ARG_KEY = "esm_data";

    private ESMFactory esmFactory = new ESMFactory();

    private ESM_Question esmQuestion;

    public static ESM_Question_Fragment newInstance(JSONObject esmJson) {
        ESM_Question_Fragment fragment = new ESM_Question_Fragment();
        Bundle args = new Bundle();
        args.putString(ESM_ARG_KEY, esmJson.toString());
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        View rootView = inflater.inflate(R.layout.esm_question_fragment, container, false);
        String esmData = getArguments().getString(ESM_ARG_KEY);

        if (esmQuestion == null) {
            JSONObject esmJson;
            try {
                esmJson = new JSONObject(esmData);
                Integer esmId = esmJson.getInt(ESM_Provider.ESM_Data._ID);
                esmJson.remove(ESM_Provider.ESM_Data._ID);
                esmQuestion = esmFactory.getESM(esmJson.getInt(ESM_Question.esm_type), esmJson, esmId);
                getChildFragmentManager().beginTransaction()
                        .replace(R.id.fragment_container, esmQuestion, ESM.TAG + esmId)
                        .commit();
            } catch (JSONException e) {
                throw new RuntimeException(e);
            }
        }

        return rootView;
    }
}
