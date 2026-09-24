package com.vigilvoice.mobile

import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object RetrofitClient {

    private const val BASE_URL = "http://192.168.21.210:8000/"
    private val client =
        OkHttpClient.Builder()

            .connectTimeout(
                30,
                TimeUnit.SECONDS
            )

            .readTimeout(
                120,
                TimeUnit.SECONDS
            )

            .writeTimeout(
                120,
                TimeUnit.SECONDS
            )

            .build()


    val api: ApiService by lazy {

        Retrofit.Builder()

            .baseUrl(
                BASE_URL
            )

            .client(
                client
            )

            .addConverterFactory(
                GsonConverterFactory.create()
            )

            .build()

            .create(
                ApiService::class.java
            )
    }
}