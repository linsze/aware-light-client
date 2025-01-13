package com.aware.ui;

import android.app.NotificationManager;
import android.content.*;
import android.database.Cursor;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;
import androidx.lifecycle.ViewModelProvider;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;

import com.aware.Aware;
import com.aware.Aware_Preferences;
import com.aware.ESM;
import com.aware.R;
import com.aware.providers.ESM_Provider;
import com.aware.providers.ESM_Provider.ESM_Data;
import com.aware.ui.esms.ESMFactory;
import com.aware.ui.esms.ESM_Question;

import org.json.JSONException;
import org.json.JSONObject;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Processes an  ESM queue until it's over.
 *
 * @author denzilferreira
 */
public class ESM_Queue extends FragmentActivity {

    private static String TAG = "AWARE::ESM Queue";

    public ESM_State esmStateListener = new ESM_State();

    private static ArrayList<JSONObject> esmJSONList = new ArrayList<>();
    private static ArrayList<ESM_Question> esmQuestions = new ArrayList<>();
    private static ESMAdapter esmAdapter;
    private SharedViewModel sharedViewModel;

    private static Button prevButton;

    private static Button nextButton;

    private static ESMFactory esmFactory = new ESMFactory();

    private static ViewPager2 viewPager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        //Clear notification if it exists, since we are going through the ESMs
        NotificationManager manager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        manager.cancel(ESM.ESM_NOTIFICATION_ID);

        TAG = Aware.getSetting(getApplicationContext(), Aware_Preferences.DEBUG_TAG).length() > 0 ? Aware.getSetting(getApplicationContext(), Aware_Preferences.DEBUG_TAG) : TAG;

        Intent queue_started = new Intent(ESM.ACTION_AWARE_ESM_QUEUE_STARTED);
        sendBroadcast(queue_started);

        IntentFilter filter = new IntentFilter();
        filter.addAction(ESM.ACTION_AWARE_ESM_QUEUE_COMPLETE);
        filter.addAction(ESM.ACTION_AWARE_ESM_EXPIRED);
        filter.addAction(ESM.ACTION_AWARE_ESM_REPLACED);
        filter.addAction(ESM.ACTION_AWARE_ESM_QUEUE_UPDATED);
        registerReceiver(esmStateListener, filter);

        sharedViewModel = new ViewModelProvider(this).get(SharedViewModel.class);
        esmJSONList = new ArrayList<>();
        esmQuestions = new ArrayList<>();
        esmAdapter = new ESMAdapter(this, esmQuestions);
        initializeQueue();

        setContentView(R.layout.esm_base);
        viewPager = findViewById(R.id.viewPager);
        viewPager.setAdapter(esmAdapter);
        viewPager.setUserInputEnabled(false);
        viewPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
            @Override
            public void onPageSelected(int position) {
                super.onPageSelected(position);
                updateButtonStates(position);
            }
        });

        prevButton = findViewById(R.id.prevButton);
        nextButton = findViewById(R.id.nextButton);

        prevButton.setOnClickListener(v -> {
            int currentItem = viewPager.getCurrentItem();
            ESM_Question esm = esmQuestions.get(currentItem);
            if (esm != null) {
                esm.saveData();
            }
            if (currentItem > 0) {
                viewPager.setCurrentItem(currentItem - 1);
            }
        });

        nextButton.setOnClickListener(v -> {
            int currentItem = viewPager.getCurrentItem();
            ESM_Question currentEsm = esmQuestions.get(currentItem);
            if (currentEsm != null) {
                currentEsm.saveData();
            }
            if (currentItem < esmQuestions.size() - 1) {
                viewPager.setCurrentItem(currentItem + 1);
            } else if (currentItem == esmQuestions.size() - 1) {
                Intent submittedEsm = new Intent(ESM.ACTION_AWARE_ESM_SUBMITTED);
                sendBroadcast(submittedEsm);
            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();

        if (ESM.isESMVisible(getApplicationContext())) {
            if (Aware.DEBUG)
                Log.d(Aware.TAG, "ESM was visible but not answered, go back to notification bar");

            //Revert to NEW state
            for (JSONObject esmJSON: esmJSONList) {
                try {
                    Integer esmId = esmJSON.getInt(ESM_Data._ID);
                    ContentValues rowData = new ContentValues();
                    rowData.put(ESM_Data.ANSWER_TIMESTAMP, 0);
                    rowData.put(ESM_Data.STATUS, ESM.STATUS_NEW);
                    getContentResolver().update(ESM_Data.CONTENT_URI, rowData, ESM_Data._ID + "=" + esmId, null);
                } catch (JSONException e) {
                    throw new RuntimeException(e);
                }
            }
            ESM.notifyESM(getApplicationContext(), true);
            finish();
        }
    }

    private void updateButtonStates(int position) {
        prevButton.setEnabled(position > 0);
        if (position == esmQuestions.size()-1) {
            nextButton.setText("Submit");
        } else {
            nextButton.setText("Next");
        }
    }


    public void initializeQueue() {
        try {
            Cursor current_esm;
            current_esm = getContentResolver().query(ESM_Data.CONTENT_URI, null, ESM_Data.STATUS + "=" + ESM.STATUS_NEW, null, ESM_Data.TIMESTAMP + " ASC");
            if (current_esm != null && current_esm.moveToFirst()) {
                do {
                    int _id = current_esm.getInt(current_esm.getColumnIndex(ESM_Data._ID));

                    //Fixed: set the esm as VISIBLE, to avoid displaying the same ESM twice due to changes in orientation
                    ContentValues update_state = new ContentValues();
                    update_state.put(ESM_Data.STATUS, ESM.STATUS_VISIBLE);
                    getContentResolver().update(ESM_Data.CONTENT_URI, update_state, ESM_Data._ID + "=" + _id, null);

                    //Load esm question JSON from database
                    JSONObject esm_question = new JSONObject(current_esm.getString(current_esm.getColumnIndex(ESM_Data.JSON)));
                    esm_question = esm_question.put(ESM_Data._ID, current_esm.getInt(current_esm.getColumnIndex(ESM_Data._ID)));
                    esmJSONList.add(esm_question);
                    ESM_Question esmQuestion = esmFactory.getESM(esm_question.getInt(ESM_Question.esm_type), esm_question, current_esm.getInt(current_esm.getColumnIndex(ESM_Data._ID)));
                    esmQuestions.add(esmQuestion);
                } while (current_esm.moveToNext());
            }
            if (current_esm != null && !current_esm.isClosed()) current_esm.close();

            // There may be ESM to remove due to change of answer
            Cursor branched_esm = getContentResolver().query(ESM_Data.CONTENT_URI, null, ESM_Data.STATUS + "=" + ESM.STATUS_BRANCHED, null, ESM_Data.TIMESTAMP + " ASC");
            if (branched_esm != null && branched_esm.moveToFirst()) {
                do {
                    //Load esm question JSON from database
                    JSONObject esm_question = new JSONObject(branched_esm.getString(branched_esm.getColumnIndex(ESM_Data.JSON)));
                    esm_question = esm_question.put(ESM_Data._ID, branched_esm.getInt(branched_esm.getColumnIndex(ESM_Data._ID)));
                    for (int i=0; i<esmJSONList.size(); i++) {
//                        if (esmJSONList.get(i).toString().equals(esm_question.toString())) {
                        if (esm_question.get("esm_title").equals(esmJSONList.get(i).get("esm_title")) &&
                                esm_question.get("_id").equals(esmJSONList.get(i).get("_id"))) {
                            esmJSONList.remove(i);
                            esmQuestions.remove(i);
                        }
                    }
                } while (branched_esm.moveToNext());
            }
            if (branched_esm != null && !branched_esm.isClosed()) branched_esm.close();

            esmAdapter.notifyDataSetChanged();
        } catch (JSONException e) {
            e.printStackTrace();
        }
        if (esmAdapter.getItemCount() == 0) {
            finish();
        }
    }

    public class ESM_State extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (intent.getAction().equals(ESM.ACTION_AWARE_ESM_QUEUE_COMPLETE)) {
                //Clean-up trials from database
                getContentResolver().delete(ESM_Data.CONTENT_URI, ESM_Data.TRIGGER + " LIKE 'TRIAL'", null);
                ExecutorService executorService = Executors.newSingleThreadExecutor(); // Use single thread for sequential execution
                // Executes data saving in the background
                executorService.execute(() -> {
                    for (ESM_Question esm : esmQuestions) {
                        ContentValues rowData = new ContentValues();
                        rowData.put(ESM_Provider.ESM_Data.STATUS, ESM.STATUS_SUBMITTED);
                        getContentResolver().update(ESM_Provider.ESM_Data.CONTENT_URI, rowData, ESM_Provider.ESM_Data._ID + "=" + esm.getID(), null);
                    }
                    runOnUiThread(() -> {
                        Toast.makeText(getApplicationContext(), "Responses submitted", Toast.LENGTH_LONG).show();
                        finish();
                    });
                });
                executorService.shutdown();
            } else if (intent.getAction().equals(ESM.ACTION_AWARE_ESM_QUEUE_UPDATED)) {
                initializeQueue();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        unregisterReceiver(esmStateListener);
    }

    /**
     * Get amount of ESMs waiting on database (visible or new)
     *
     * @return int count
     */
    public static int getQueueSize(Context c) {
        int size = 0;
        Cursor onqueue = c.getContentResolver().query(ESM_Data.CONTENT_URI, null, ESM_Data.STATUS + " IN (" + ESM.STATUS_VISIBLE + "," + ESM.STATUS_NEW + ")", null, null);
        if (onqueue != null && onqueue.moveToFirst()) {
            size = onqueue.getCount();
        }
        if (onqueue != null && !onqueue.isClosed()) onqueue.close();

        Log.d(TAG, "Queue size: " + size);
        return size;
    }

    /**
     * Get dialog timeout value. How long is the ESM visible on the screen
     * @param c
     * @return
     */
    public static int getExpirationThreshold(Context c) {
        int expiration = 0;
        String[] projection = { ESM_Data.EXPIRATION_THRESHOLD };
        Cursor onqueue = c.getContentResolver().query(ESM_Data.CONTENT_URI, projection, ESM_Data.STATUS + "=" + ESM.STATUS_VISIBLE, null, null);
        if (onqueue != null && onqueue.moveToFirst()) {
            expiration = onqueue.getInt(onqueue.getColumnIndex(ESM_Data.EXPIRATION_THRESHOLD));
        }
        if (onqueue != null && !onqueue.isClosed()) onqueue.close();
        return expiration;
    }

    /**
     * Get notification timeout value. How long is the ESM notification visible on the tray
     * @param c
     * @return
     */
    public static int getNotificationTimeout(Context c) {
        int timeout = 0;
        String[] projection = { ESM_Data.NOTIFICATION_TIMEOUT };
        Cursor onqueue = c.getContentResolver().query(ESM_Data.CONTENT_URI, projection, ESM_Data.STATUS + "=" + ESM.STATUS_NEW, null, null);
        if (onqueue != null && onqueue.moveToFirst()) {
            timeout = onqueue.getInt(onqueue.getColumnIndex(ESM_Data.NOTIFICATION_TIMEOUT));
        }
        if (onqueue != null && !onqueue.isClosed()) onqueue.close();
        return timeout;
    }

    /**
     * A ViewModel shared across ESM fragments to restore data across navigation.
     * Each ESM fragment will listen to live data changes using respective IDs as keys.
     */
    public static class SharedViewModel extends ViewModel {
        private Map<Integer, MutableLiveData<Object>> esmData = new HashMap<>();
        public SharedViewModel() {}

        public void storeData(Integer key, Object data) {
            MutableLiveData<Object> liveData = esmData.get(key);
            if (liveData == null) {
                liveData = new MutableLiveData<>();
                esmData.put(key, liveData);
            }
            liveData.setValue(data);
        }

        public LiveData<Object> getStoredData(Integer key) {
            if (!esmData.containsKey(key)) {
                esmData.put(key, new MutableLiveData<>());
            }
            return esmData.get(key);
        }
    }

    public SharedViewModel getSharedViewModel() {
        return sharedViewModel;
    }

    public class ESMAdapter extends FragmentStateAdapter {
        private ArrayList<ESM_Question> esmQuestions;

        public ESMAdapter(@NonNull FragmentActivity fragmentActivity, ArrayList<ESM_Question> esmQuestions) {
            super(fragmentActivity);
            this.esmQuestions = esmQuestions;
        }

        @NonNull
        @Override
        public Fragment createFragment(int position) {
            return esmQuestions.get(position);
        }

        @Override
        public int getItemCount() {
            return esmQuestions.size();
        }
    }
}