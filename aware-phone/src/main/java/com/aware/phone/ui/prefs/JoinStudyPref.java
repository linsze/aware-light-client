package com.aware.phone.ui.prefs;

import static androidx.core.content.ContextCompat.getSystemService;

import android.app.Activity;
import android.content.Context;
import android.preference.Preference;
import android.util.AttributeSet;
import android.view.Display;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;

import com.aware.phone.R;
import com.aware.phone.ui.dialogs.JoinStudyDialog;

public class JoinStudyPref extends Preference {

    public JoinStudyPref(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public JoinStudyPref(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public JoinStudyPref(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public JoinStudyPref(Context context) {
        super(context);
    }

    @Override
    protected View onCreateView(ViewGroup parent) {
        super.onCreateView(parent);
        LayoutInflater inflater = (LayoutInflater) getContext().getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        View view = inflater.inflate(R.layout.btn_join_study, parent, false);
        //HACK: Referenced from https://stackoverflow.com/questions/7506230/set-position-size-of-ui-element-as-percentage-of-screen-size
        Button joinStudyButton = view.findViewById(R.id.btn_join_study);
        ViewGroup.MarginLayoutParams layoutParams = (ViewGroup.MarginLayoutParams) joinStudyButton
                .getLayoutParams();
        Display display = ((WindowManager)getContext().getSystemService(Context.WINDOW_SERVICE)).getDefaultDisplay();
        layoutParams.setMargins(layoutParams.leftMargin, ((int)(display.getHeight()*0.35)), layoutParams.rightMargin, layoutParams.bottomMargin);

        joinStudyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new JoinStudyDialog((Activity) getContext()).showDialog();
            }
        });

        return view;
    }
}
