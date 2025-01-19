package com.aware;

import static com.aware.Applications.isSystemPackage;

import android.app.AppOpsManager;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.usage.UsageEvents;
import android.app.usage.UsageStats;
import android.app.usage.UsageStatsManager;
import android.content.BroadcastReceiver;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SyncRequest;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.provider.Settings;
import android.util.Log;

import androidx.core.app.NotificationCompat;

import com.aware.providers.ApplicationUsage_Provider;
import com.aware.utils.Aware_Sensor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.aware.providers.ApplicationUsage_Provider.ApplicationUsageStats;

/**
 * Created by Lin Sze Khoo
 * Queries package usage events based on screen off events broadcasted by device_usage plugin
 * Require usage access to be enabled manually (separated from typical permission access) but does not require accessibility
 */
public class ApplicationUsage extends Aware_Sensor {

//    public static HashMap<String, HashMap<String, String>> SETTINGS_PERMISSIONS = new HashMap<String, HashMap<String, String>>(){{
//        put(Manifest.permission.PACKAGE_USAGE_STATS, new HashMap<String, String>(){{
//            put("Application usage", Aware_Preferences.STATUS_APPLICATION_USAGE);
//        }});
//    }};
//
//    private static ArrayList<String> ADDITIONAL_PERMISSIONS = new ArrayList<String>(){{
//        add(Manifest.permission.PACKAGE_USAGE_STATS);
//    }};

    public static String TAG = "AWARE::Application usage";

    public static final int USAGE_ACCESS_NOTIFICATION_ID = 43;

    private static UsageStatsManager usageStatsManager;

    private static SyncAppUsageReceiver syncAppUsageReceiver;

    private PackageManager packageManager;
    @Override
    public void onCreate() {
        super.onCreate();
        AUTHORITY = ApplicationUsage_Provider.getAuthority(this);

        syncAppUsageReceiver = new SyncAppUsageReceiver();
        IntentFilter filter = new IntentFilter();
        filter.addAction(Aware.ACTION_AWARE_SYNC_DATA);
        filter.addAction(Screen.ACTION_AWARE_PLUGIN_DEVICE_USAGE);
        registerReceiver(syncAppUsageReceiver, filter);

        usageStatsManager = (UsageStatsManager) getSystemService(Context.USAGE_STATS_SERVICE);
        packageManager = getPackageManager();
        if (Aware.DEBUG) Log.d(TAG, "Application usage service created!");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        super.onStartCommand(intent, flags, startId);

        if (!isUsageAccessEnabled()) {
            stopSelf();
            return START_NOT_STICKY;
        }

        if (PERMISSIONS_OK) {
            DEBUG = Aware.getSetting(this, Aware_Preferences.DEBUG_FLAG).equals("true");
            Aware.setSetting(this, Aware_Preferences.STATUS_APPLICATION_USAGE, true);

            if (Aware.DEBUG) Log.d(TAG, "Application usage active...");

            if (Aware.isStudy(this)) {
                ContentResolver.setIsSyncable(Aware.getAWAREAccount(this), ApplicationUsage_Provider.getAuthority(this), 1);
                ContentResolver.setSyncAutomatically(Aware.getAWAREAccount(this), ApplicationUsage_Provider.getAuthority(this), true);
                long frequency;
                try {
                    frequency = Long.parseLong(Aware.getSetting(this, Aware_Preferences.FREQUENCY_WEBSERVICE)) * 60;
                } catch (NumberFormatException e) {
                    frequency = 30 * 60;
                }
                SyncRequest request = new SyncRequest.Builder()
                        .syncPeriodic(frequency, frequency / 3)
                        .setSyncAdapter(Aware.getAWAREAccount(this), ApplicationUsage_Provider.getAuthority(this))
                        .setExtras(new Bundle()).build();
                ContentResolver.requestSync(request);
            }
        } else {
            stopSelf();
            return START_NOT_STICKY;
        }
        return START_STICKY;
    }

    private boolean isUsageAccessEnabled() {
        AppOpsManager appOps = (AppOpsManager) getSystemService(Context.APP_OPS_SERVICE);
        int mode = appOps.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, android.os.Process.myUid(), getPackageName());
        boolean granted = (mode == AppOpsManager.MODE_ALLOWED);
        return granted;
    }

    public static synchronized void promptUsageAccessNotification(Context c) {
        NotificationCompat.Builder mBuilder = new NotificationCompat.Builder(c, Aware.AWARE_NOTIFICATION_CHANNEL_GENERAL);
        mBuilder.setSmallIcon(R.drawable.ic_stat_aware_accessibility);
        mBuilder.setContentTitle(c.getResources().getString(R.string.aware_allow_usage_access_title));
        mBuilder.setContentText(c.getResources().getString(R.string.aware_allow_usage_access));
        mBuilder.setAutoCancel(true);
        mBuilder.setOnlyAlertOnce(true); //notify the user only once
        mBuilder.setDefaults(NotificationCompat.DEFAULT_ALL);
        mBuilder = Aware.setNotificationProperties(mBuilder, Aware.AWARE_NOTIFICATION_IMPORTANCE_GENERAL);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
            mBuilder.setChannelId(Aware.AWARE_NOTIFICATION_CHANNEL_GENERAL);

        Intent intentToSettings = new Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS);
        intentToSettings.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);

        PendingIntent clickIntent = PendingIntent.getActivity(c, 0, intentToSettings, PendingIntent.FLAG_UPDATE_CURRENT);
        mBuilder.setContentIntent(clickIntent);

        NotificationManager notManager = (NotificationManager) c.getSystemService(Context.NOTIFICATION_SERVICE);
        notManager.notify(USAGE_ACCESS_NOTIFICATION_ID, mBuilder.build());
    }

    public class SyncAppUsageReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (intent.getAction().equals(Aware.ACTION_AWARE_SYNC_DATA)) {
                Bundle sync = new Bundle();
                sync.putBoolean(ContentResolver.SYNC_EXTRAS_MANUAL, true);
                sync.putBoolean(ContentResolver.SYNC_EXTRAS_EXPEDITED, true);

                ContentResolver.requestSync(Aware.getAWAREAccount(context), ApplicationUsage_Provider.AUTHORITY, sync);
            }

            if (intent.getAction().equals(Screen.ACTION_AWARE_PLUGIN_DEVICE_USAGE)) {
                long elapsedDeviceOn = intent.getLongExtra(Screen.EXTRA_ELAPSED_DEVICE_ON, 0);
                if (elapsedDeviceOn > 0) {
                    List<ContentValues> appUsageEntries = new ArrayList<>();
                    long screenOffTimestamp = System.currentTimeMillis();
                    long screenOnTimestamp = screenOffTimestamp - elapsedDeviceOn;
//                    List<UsageStats> usageStats = usageStatsManager.queryUsageStats(
//                            UsageStatsManager.INTERVAL_DAILY, screenOnTimestamp, screenOffTimestamp);
//                    if (usageStats != null && usageStats.size() > 0) {
//                        for (UsageStats usageStat : usageStats) {
//                            long usageDuration = usageStat.getTotalTimeInForeground();
//                            if (usageDuration > 0) {
//                                String packageName = usageStat.getPackageName().toString();
//                                long startTimestamp = usageStat.getFirstTimeStamp();
//                                ContentValues rowData = constructAppUsageEntry(packageName, startTimestamp, startTimestamp + usageDuration);
//                                getContentResolver().insert(ApplicationUsageStats.CONTENT_URI, rowData);
//                            }
//                        }
//                    }
                    Map<String, Long> appForegroundDurations = new HashMap<>();
                    UsageEvents usageEvents = usageStatsManager.queryEvents(screenOnTimestamp, screenOffTimestamp);
                    UsageEvents.Event event = new UsageEvents.Event();

                    while (usageEvents.hasNextEvent()) {
                        usageEvents.getNextEvent(event);
                        // Ignore events outside the time range or for system processes
                        if (event.getTimeStamp() < screenOnTimestamp || event.getTimeStamp() > screenOffTimestamp) {
                            continue;
                        }
                        String packageName = event.getPackageName();
                        if (!appForegroundDurations.containsKey(packageName)) {
                            appForegroundDurations.put(packageName, 0L);
                        }

                        if (event.getEventType() == UsageEvents.Event.MOVE_TO_FOREGROUND) {
                            appForegroundDurations.put(packageName + "_start", event.getTimeStamp());
                        } else if (event.getEventType() == UsageEvents.Event.MOVE_TO_BACKGROUND) {
                            if (appForegroundDurations.containsKey(packageName + "_start")) {
                                long foregroundStartTime = appForegroundDurations.get(packageName + "_start");
                                ContentValues rowData = constructAppUsageEntry(packageName, foregroundStartTime, event.getTimeStamp());
                                appUsageEntries.add(rowData);
                                appForegroundDurations.remove(packageName + "_start");
                            }
                        }
                    }

                    // Edge case where the app is still in the foreground when the interval ends
                    for (Map.Entry<String, Long> packageEntry : appForegroundDurations.entrySet()) {
                        if (packageEntry.getKey().endsWith("_start")) {
                            String packageName = packageEntry.getKey().replace("_start", "");
                            ContentValues rowData = constructAppUsageEntry(packageName, screenOnTimestamp, screenOffTimestamp);
                            appUsageEntries.add(rowData);
                        }
                    }

                    if (appUsageEntries.size() > 0) {
                        ContentValues[] contentValuesArray = new ContentValues[appUsageEntries.size()];
                        contentValuesArray = appUsageEntries.toArray(contentValuesArray);
                        getContentResolver().bulkInsert(ApplicationUsageStats.CONTENT_URI, contentValuesArray);
                    }
                }
            }
        }
    }

    private ContentValues constructAppUsageEntry(String packageName, long startTimestamp, long endTimestamp) {
        ApplicationInfo appInfo;
        try {
            appInfo = packageManager.getApplicationInfo(packageName, PackageManager.GET_META_DATA);
        } catch (final PackageManager.NameNotFoundException e) {
            appInfo = null;
        }
        String appName = "";
        if (appInfo != null && packageManager.getApplicationLabel(appInfo) != null) {
            appName = (String) packageManager.getApplicationLabel(appInfo);
        }

        PackageInfo packageInfo;
        try {
            packageInfo = packageManager.getPackageInfo(packageName, PackageManager.GET_META_DATA);
        } catch (final PackageManager.NameNotFoundException e) {
            packageInfo = null;
        }

        ContentValues rowData = new ContentValues();
        rowData.put(ApplicationUsageStats.TIMESTAMP, System.currentTimeMillis());
        rowData.put(ApplicationUsageStats.START_TIMESTAMP, startTimestamp);
        rowData.put(ApplicationUsageStats.END_TIMESTAMP, endTimestamp);
        rowData.put(ApplicationUsageStats.FOREGROUND_DURATION, endTimestamp - startTimestamp);
        rowData.put(ApplicationUsageStats.DEVICE_ID, Aware.getSetting(getApplicationContext(), Aware_Preferences.DEVICE_ID));
        rowData.put(ApplicationUsageStats.PACKAGE_NAME, packageName);
        rowData.put(ApplicationUsageStats.APPLICATION_NAME, appName);
        rowData.put(ApplicationUsageStats.IS_SYSTEM_APP, packageInfo != null && isSystemPackage(packageInfo));
        return rowData;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        unregisterReceiver(syncAppUsageReceiver);
        ContentResolver.setSyncAutomatically(Aware.getAWAREAccount(this), ApplicationUsage_Provider.getAuthority(this), false);
        ContentResolver.removePeriodicSync(
                Aware.getAWAREAccount(this),
                ApplicationUsage_Provider.getAuthority(this),
                Bundle.EMPTY
        );

        if (Aware.DEBUG) Log.d(TAG, "Application usage service terminated...");
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
