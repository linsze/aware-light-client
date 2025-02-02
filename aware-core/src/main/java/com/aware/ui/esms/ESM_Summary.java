package com.aware.ui.esms;

import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.method.ScrollingMovementMethod;
import android.text.style.ForegroundColorSpan;
import android.text.style.StyleSpan;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import com.aware.R;

import org.json.JSONException;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Display a summary of all ESM responses in the current queue.
 */
public class ESM_Summary extends ESM_Question {
    private TextView summaryText;
    private ArrayList<ESM_Question> esmQuestions;

    public ESM_Summary(ArrayList<ESM_Question> esmQuestions) {
        this.esmQuestions = esmQuestions;
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.esm_summary, container, false);
    }

    @Nullable
    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        summaryText = view.findViewById(R.id.summary_text);
        summaryText.setMovementMethod(new ScrollingMovementMethod());

        sharedViewModel.getAllAnswers().observe(getViewLifecycleOwner(), answers -> {
            SpannableStringBuilder spannableSummary = new SpannableStringBuilder();
            try {
                for (ESM_Question esm: esmQuestions) {
                    Integer esmID = esm.getID();
                    Object answer = answers.get(esmID);
                    if (answer != null) {
                        String titleText = esm.getTitle();
                        SpannableString titleSpannable = new SpannableString(titleText);
                        titleSpannable.setSpan(new StyleSpan(Typeface.BOLD), 0, titleText.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
                        spannableSummary.append(titleSpannable);

                        String answerText = "\nAnswer: " + answer.toString() + "\n\n";
                        SpannableString answerSpannable = new SpannableString(answerText);
                        answerSpannable.setSpan(new StyleSpan(Typeface.ITALIC), 0, answerText.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
                        answerSpannable.setSpan(new ForegroundColorSpan(Color.GRAY), 0, answerText.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
                        spannableSummary.append(answerSpannable);
                    }

                }
                summaryText.setText(spannableSummary);
            } catch (JSONException e) {
                e.printStackTrace();
            }
        });
    }
}
