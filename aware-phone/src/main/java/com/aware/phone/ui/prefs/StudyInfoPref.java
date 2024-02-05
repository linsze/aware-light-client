package com.aware.phone.ui.prefs;

import android.content.Context;
import android.database.Cursor;
import android.graphics.Typeface;
import android.preference.Preference;
import android.text.Html;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.BulletSpan;
import android.text.style.StyleSpan;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.phone.R;
import com.aware.providers.Aware_Provider;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StudyInfoPref extends Preference {

    public StudyInfoPref(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public StudyInfoPref(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public StudyInfoPref(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public StudyInfoPref(Context context) {
        super(context);
    }

    @Override
    protected View onCreateView(ViewGroup parent) {
        super.onCreateView(parent);
        LayoutInflater inflater = (LayoutInflater) getContext().getSystemService(
                Context.LAYOUT_INFLATER_SERVICE);
        View view = inflater.inflate(R.layout.pref_study_info, parent, false);

        TextView tvStudyName = view.findViewById(R.id.study_name);
        TextView tvStudyDesc = view.findViewById(R.id.study_description);
        TextView tvStudyContact = view.findViewById(R.id.study_contact);

        Cursor study = Aware.getStudy(getContext(),
                Aware.getSetting(getContext(), Aware_Preferences.WEBSERVICE_SERVER));
        if (study != null && study.moveToFirst()) {
            tvStudyName.setText(study.getString(
                    study.getColumnIndex(Aware_Provider.Aware_Studies.STUDY_TITLE)));
            tvStudyDesc.setText(Html.fromHtml(study.getString(study.getColumnIndex(
                    Aware_Provider.Aware_Studies.STUDY_DESCRIPTION)), null, null));
            tvStudyContact.setText(study.getString(study.getColumnIndex(
                    Aware_Provider.Aware_Studies.STUDY_PI)));
        }

        // Initialize bullet point description items
        HashMap<Integer, Integer> appNavigationContent = new HashMap<>();
        appNavigationContent.put(R.id.app_features_bullet, R.string.app_features);
        appNavigationContent.put(R.id.app_user_control_bullet, R.string.app_user_control);
        appNavigationContent.put(R.id.app_future_features_bullet, R.string.app_future_features);

        List<String> keywordsToBold = new ArrayList<String>(){{
            add("Settings");
            add("Data");
        }};
        for (int textViewId: appNavigationContent.keySet()) {
            TextView contentHolder = view.findViewById(textViewId);
            String featureString = getContext().getString(appNavigationContent.get(textViewId));
            SpannableStringBuilder sbBuilder = new SpannableStringBuilder();
            String[] bulletItems = featureString.split("\n");
            for (int i = 0; i < bulletItems.length; i++) {
                String currentItem = bulletItems[i];
                Spannable itemSpan = new SpannableString(currentItem + (i < bulletItems.length-1 ? "\n" : ""));
                itemSpan.setSpan(new BulletSpan(15), 0, itemSpan.length(), Spanned.SPAN_INCLUSIVE_EXCLUSIVE);
                for (String keyword: keywordsToBold) {
                    if (currentItem.contains(keyword)) {
                        int keywordIndex = currentItem.lastIndexOf(keyword);
                        itemSpan.setSpan(new StyleSpan(Typeface.BOLD_ITALIC), keywordIndex, keywordIndex+keyword.length(), 0);
                    }
                }
                sbBuilder.append(itemSpan);
            }
            contentHolder.setText(sbBuilder);
        }

        return view;
    }
}
