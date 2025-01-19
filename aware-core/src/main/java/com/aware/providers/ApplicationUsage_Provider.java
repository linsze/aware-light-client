package com.aware.providers;

import android.content.ContentProvider;
import android.content.ContentUris;
import android.content.ContentValues;
import android.content.Context;
import android.content.UriMatcher;
import android.database.Cursor;
import android.database.SQLException;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteQueryBuilder;
import android.net.Uri;
import android.provider.BaseColumns;
import android.util.Log;

import com.aware.Aware;
import com.aware.utils.DatabaseHelper;

import java.util.HashMap;

/**
 * Created by Lin Sze Khoo
 */
public class ApplicationUsage_Provider extends ContentProvider {

    public static final int DATABASE_VERSION = 7;
    /**
     * Authority of content provider
     */
    public static String AUTHORITY = "com.aware.provider.application.usage";

    // ContentProvider query paths
    private final int APP_USAGE = 1;
    private final int APP_USAGE_ID = 2;

    /**
     * Applications running on the foreground
     */
    public static final class ApplicationUsageStats implements BaseColumns {
        private ApplicationUsageStats() {
        }

        public static final Uri CONTENT_URI = Uri.parse("content://" + ApplicationUsage_Provider.AUTHORITY + "/application_usage");
        public static final String CONTENT_TYPE = "vnd.android.cursor.dir/vnd.aware.application.usage";
        public static final String CONTENT_ITEM_TYPE = "vnd.android.cursor.item/vnd.aware.application.usage";

        public static final String _ID = "_id";
        public static final String TIMESTAMP = "timestamp";
        public static final String START_TIMESTAMP = "start_timestamp";
        public static final String END_TIMESTAMP = "end_timestamp";
        public static final String FOREGROUND_DURATION = "foreground_duration";
        public static final String DEVICE_ID = "device_id";
        public static final String PACKAGE_NAME = "package_name";
        public static final String APPLICATION_NAME = "application_name";
        public static final String IS_SYSTEM_APP = "is_system_app";
    }

    public static String DATABASE_NAME = "application_usage.db";

    public static final String[] DATABASE_TABLES = {"application_usage"};
    public static final String[] TABLES_FIELDS = {
            ApplicationUsage_Provider.ApplicationUsageStats._ID + " integer primary key autoincrement,"
                    + ApplicationUsage_Provider.ApplicationUsageStats.TIMESTAMP + " real default 0,"
                    + ApplicationUsage_Provider.ApplicationUsageStats.START_TIMESTAMP + " real default 0,"
                    + ApplicationUsage_Provider.ApplicationUsageStats.END_TIMESTAMP + " real default 0,"
                    + ApplicationUsage_Provider.ApplicationUsageStats.FOREGROUND_DURATION + " real default 0,"
                    + ApplicationUsage_Provider.ApplicationUsageStats.DEVICE_ID + " text default '',"
                    + ApplicationUsage_Provider.ApplicationUsageStats.PACKAGE_NAME + " text default '',"
                    + ApplicationUsage_Provider.ApplicationUsageStats.APPLICATION_NAME + " text default '',"
                    + ApplicationUsage_Provider.ApplicationUsageStats.IS_SYSTEM_APP + " integer default 0"};

    private UriMatcher sUriMatcher = null;
    private HashMap<String, String> appUsageMap = null;

    private DatabaseHelper dbHelper;
    private static SQLiteDatabase database;

    private void initialiseDatabase() {
        if (dbHelper == null)
            dbHelper = new DatabaseHelper(getContext(), DATABASE_NAME, null, DATABASE_VERSION, DATABASE_TABLES, TABLES_FIELDS);
        if (database == null)
            database = dbHelper.getWritableDatabase();
    }

    /**
     * Delete entry from the database
     */
    @Override
    public synchronized int delete(Uri uri, String selection, String[] selectionArgs) {

        initialiseDatabase();

        //lock database for transaction
        database.beginTransaction();

        int count;
        switch (sUriMatcher.match(uri)) {
            case APP_USAGE:
                count = database.delete(DATABASE_TABLES[0], selection, selectionArgs);
                break;
            default:
                database.endTransaction();
                throw new IllegalArgumentException("Unknown URI " + uri);
        }

        database.setTransactionSuccessful();
        database.endTransaction();

        getContext().getContentResolver().notifyChange(uri, null, false);

        return count;
    }

    @Override
    public String getType(Uri uri) {
        switch (sUriMatcher.match(uri)) {
            case APP_USAGE:
                return ApplicationUsage_Provider.ApplicationUsageStats.CONTENT_TYPE;
            case APP_USAGE_ID:
                return ApplicationUsage_Provider.ApplicationUsageStats.CONTENT_ITEM_TYPE;
            default:
                throw new IllegalArgumentException("Unknown URI " + uri);
        }
    }

    /**
     * Insert entry to the database
     */
    @Override
    public synchronized Uri insert(Uri uri, ContentValues initialValues) {

        initialiseDatabase();

        ContentValues values = (initialValues != null) ? new ContentValues(initialValues) : new ContentValues();

        database.beginTransaction();

        switch (sUriMatcher.match(uri)) {
            case APP_USAGE:
                long app_usage_id = database.insertWithOnConflict(DATABASE_TABLES[0], ApplicationUsage_Provider.ApplicationUsageStats.APPLICATION_NAME, values, SQLiteDatabase.CONFLICT_IGNORE);
                if (app_usage_id > 0) {
                    Uri appUsageUri = ContentUris.withAppendedId(ApplicationUsage_Provider.ApplicationUsageStats.CONTENT_URI, app_usage_id);
                    getContext().getContentResolver().notifyChange(appUsageUri, null, false);
                    database.setTransactionSuccessful();
                    database.endTransaction();
                    return appUsageUri;
                }
                database.endTransaction();
                throw new SQLException("Failed to insert row into " + uri);
            default:
                throw new IllegalArgumentException("Unknown URI " + uri);
        }
    }

    /**
     * Returns the provider authority that is dynamic
     * @return
     */
    public static String getAuthority(Context context) {
        AUTHORITY = context.getPackageName() + ".provider.application.usage";
        return AUTHORITY;
    }

    @Override
    public boolean onCreate() {
        AUTHORITY = getContext().getPackageName() + ".provider.application.usage";

        sUriMatcher = new UriMatcher(UriMatcher.NO_MATCH);
        sUriMatcher.addURI(ApplicationUsage_Provider.AUTHORITY, DATABASE_TABLES[0],
                APP_USAGE);
        sUriMatcher.addURI(ApplicationUsage_Provider.AUTHORITY, DATABASE_TABLES[0]
                + "/#", APP_USAGE_ID);

        appUsageMap = new HashMap<String, String>();
        appUsageMap.put(ApplicationUsage_Provider.ApplicationUsageStats._ID,
                ApplicationUsage_Provider.ApplicationUsageStats._ID);
        appUsageMap.put(ApplicationUsage_Provider.ApplicationUsageStats.TIMESTAMP,
                ApplicationUsage_Provider.ApplicationUsageStats.TIMESTAMP);
        appUsageMap.put(ApplicationUsage_Provider.ApplicationUsageStats.START_TIMESTAMP,
                ApplicationUsage_Provider.ApplicationUsageStats.START_TIMESTAMP);
        appUsageMap.put(ApplicationUsage_Provider.ApplicationUsageStats.END_TIMESTAMP,
                ApplicationUsage_Provider.ApplicationUsageStats.END_TIMESTAMP);
        appUsageMap.put(ApplicationUsage_Provider.ApplicationUsageStats.FOREGROUND_DURATION,
                ApplicationUsage_Provider.ApplicationUsageStats.FOREGROUND_DURATION);
        appUsageMap.put(ApplicationUsage_Provider.ApplicationUsageStats.DEVICE_ID,
                ApplicationUsage_Provider.ApplicationUsageStats.DEVICE_ID);
        appUsageMap.put(ApplicationUsage_Provider.ApplicationUsageStats.PACKAGE_NAME,
                ApplicationUsage_Provider.ApplicationUsageStats.PACKAGE_NAME);
        appUsageMap.put(ApplicationUsage_Provider.ApplicationUsageStats.APPLICATION_NAME,
                ApplicationUsage_Provider.ApplicationUsageStats.APPLICATION_NAME);
        appUsageMap.put(ApplicationUsage_Provider.ApplicationUsageStats.IS_SYSTEM_APP,
                ApplicationUsage_Provider.ApplicationUsageStats.IS_SYSTEM_APP);
        return true;
    }

    /**
     * Query entries from the database
     */
    @Override
    public Cursor query(Uri uri, String[] projection, String selection,
                        String[] selectionArgs, String sortOrder) {

        initialiseDatabase();

        SQLiteQueryBuilder qb = new SQLiteQueryBuilder();
        qb.setStrict(true);
        switch (sUriMatcher.match(uri)) {
            case APP_USAGE:
                qb.setTables(DATABASE_TABLES[0]);
                qb.setProjectionMap(appUsageMap);
                break;
            default:
                throw new IllegalArgumentException("Unknown URI " + uri);
        }
        try {
            Cursor c = qb.query(database, projection, selection, selectionArgs, null, null, sortOrder);
            c.setNotificationUri(getContext().getContentResolver(), uri);
            return c;
        } catch (IllegalStateException e) {
            if (Aware.DEBUG)
                Log.e(Aware.TAG, e.getMessage());
            return null;
        }
    }

    /**
     * Update application on the database
     */
    @Override
    public synchronized int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {

        initialiseDatabase();

        database.beginTransaction();

        int count;
        switch (sUriMatcher.match(uri)) {
            case APP_USAGE:
                count = database.update(DATABASE_TABLES[0], values, selection, selectionArgs);
                break;
            default:
                throw new IllegalArgumentException("Unknown URI " + uri);
        }

        database.setTransactionSuccessful();
        database.endTransaction();

        getContext().getContentResolver().notifyChange(uri, null, false);
        return count;
    }
}
